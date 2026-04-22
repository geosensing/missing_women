"""
Sampling plan generator for estimating proportion of women at Bangalore metro stations.

Usage:
    python metro_sampling_plan.py --seed 42 --n_stations 9 --n_days 4 --visits_per_station 2

The script:
1. Loads the station frame from station_frame.csv (editable, field-verifiable).
2. Draws a line-proportional random sample (with optional force-includes).
3. Computes travel times between stations (motorbike vs. metro, picks faster).
4. Builds ~4-hour field chunks across the specified days, sequences stations
   within each chunk by greedy nearest-neighbor to minimize travel.
5. Respects: budget, weekday/weekend balance, hard 22:00 curfew, all days active.
6. Outputs a human-readable itinerary and a clean CSV.

Seed makes the entire plan reproducible.
Station frame is external (station_frame.csv) so it can be ground-truthed by
someone who can actually walk into the stations.
"""

import argparse
import csv
import math
import random
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. LOAD STATION FRAME
# ---------------------------------------------------------------------------

def load_frame(path):
    """
    Load station frame from CSV.
    Expected columns: name, line, lat, lon, area, interchange
    interchange is blank or a comma-separated list of other lines serving the station.
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        stations = []
        for row in reader:
            stations.append({
                "name": row["name"].strip(),
                "line": row["line"].strip(),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "area": row["area"].strip(),
                "interchange": row.get("interchange", "").strip(),
            })
    return stations

# ---------------------------------------------------------------------------
# 2. TRAVEL TIME
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def travel_time(sa, sb):
    """Bangalore motorbike: ~20 km/h avg in traffic, 1.3x road factor, min 8 min."""
    dist = haversine_km(sa.lat, sa.lon, sb.lat, sb.lon)
    return max(8, round(dist * 1.3 / 20 * 60)), "motorbike"


# ---------------------------------------------------------------------------
# 3. PROTOCOL CONSTANTS
# ---------------------------------------------------------------------------

PROTOCOL = {
    "point_a": {
        "label": "Fare gate area",
        "description": "Stand 2-3m from fare gates on concourse side facing gates. "
                       "Record entering and exiting passengers.",
        "duration_min": 15,
    },
    "point_b": {
        "label": "Platform",
        "description": "Stand at platform midpoint facing along platform length. "
                       "Capture at least one train arrival/departure cycle.",
        "duration_min": 10,
    },
    "setup_transition_min": 5,
}

OBSERVATION_MIN = (
    PROTOCOL["point_a"]["duration_min"]
    + PROTOCOL["point_b"]["duration_min"]
    + PROTOCOL["setup_transition_min"]
)

# ---------------------------------------------------------------------------
# 4. SAMPLING
# ---------------------------------------------------------------------------

@dataclass
class SampledStation:
    name: str
    line: str
    lat: float
    lon: float
    area: str
    interchange: str


@dataclass
class Visit:
    station: SampledStation
    visit_number: int
    day_type_req: str      # "weekday", "weekend", or "any"
    chunk_idx: int = -1
    day: int = -1
    day_type: str = ""
    arrive_time: str = ""
    depart_time: str = ""
    travel_from: str = ""
    travel_min: int = 0
    travel_mode: str = ""


def sample_stations(stations, n, rng, force_include=None):
    """
    Line-proportional sampling.
    Force-included stations count toward their line's quota.
    Remaining slots allocated proportional to line size, filled by SRS within line.
    """
    force_include = force_include or set()
    forced = [s for s in stations if s["name"] in force_include]
    forced_names = {s["name"] for s in forced}

    # Group by line (excluding forced)
    lines = {}
    for s in stations:
        if s["name"] not in forced_names:
            lines.setdefault(s["line"], []).append(s)

    # Frame size per line
    frame_per_line = Counter(s["line"] for s in stations)
    total_frame = sum(frame_per_line.values())

    remaining_n = max(0, n - len(forced))

    # Proportional allocation
    alloc = {}
    allocated = 0
    for line in sorted(lines.keys()):
        share = round(remaining_n * frame_per_line[line] / total_frame)
        share = min(max(0, share), len(lines.get(line, [])))
        alloc[line] = share
        allocated += share

    # Fix rounding
    while allocated < remaining_n:
        for line in sorted(lines, key=lambda l: len(lines.get(l, [])), reverse=True):
            if alloc.get(line, 0) < len(lines.get(line, [])):
                alloc[line] = alloc.get(line, 0) + 1
                allocated += 1
                break
        else:
            break
    while allocated > remaining_n:
        for line in sorted(lines, key=lambda l: alloc.get(l, 0), reverse=True):
            if alloc.get(line, 0) > 0:
                alloc[line] -= 1
                allocated -= 1
                break

    chosen = []
    for line, pool in lines.items():
        chosen.extend(rng.sample(pool, alloc.get(line, 0)))

    return [
        SampledStation(
            name=s["name"], line=s["line"], lat=s["lat"], lon=s["lon"],
            area=s["area"], interchange=s.get("interchange", ""),
        )
        for s in forced + chosen
    ]


def create_visit_pool(sampled_stations, visits_per_station):
    pool = []
    for s in sampled_stations:
        for v in range(visits_per_station):
            if visits_per_station >= 2:
                req = "weekday" if v == 0 else ("weekend" if v == 1 else "any")
            else:
                req = "any"
            pool.append(Visit(station=s, visit_number=v + 1, day_type_req=req))
    return pool


# ---------------------------------------------------------------------------
# 5. SCHEDULING
# ---------------------------------------------------------------------------

CHUNK_STARTS = [
    (6, 30, "early_am"),
    (8, 0,  "am_peak"),
    (11, 0, "midday"),
    (14, 0, "afternoon"),
    (17, 0, "pm_peak"),
]

EST_STATIONS_PER_CHUNK = 5


def minutes_to_hhmm(total_min):
    h = int(total_min) // 60
    m = int(total_min) % 60
    return f"{h:02d}:{m:02d}"


def define_chunks(visit_pool, n_days, rng, budget_hours=16, chunk_hours=4,
                  weekday_days=None, weekend_days=None):
    if weekday_days is None:
        weekday_days = list(range(n_days - 1))
    if weekend_days is None:
        weekend_days = [n_days - 1]

    HARD_END = 22 * 60
    chunk_min = int(chunk_hours * 60)

    valid_starts = [(h, m, label) for h, m, label in CHUNK_STARTS
                    if h * 60 + m + chunk_min <= HARD_END]

    n_weekday = sum(1 for v in visit_pool if v.day_type_req == "weekday")
    n_weekend = sum(1 for v in visit_pool if v.day_type_req == "weekend")
    n_any = sum(1 for v in visit_pool if v.day_type_req == "any")

    wd_chunks_needed = max(1, math.ceil(n_weekday / EST_STATIONS_PER_CHUNK))
    we_chunks_needed = max(1, math.ceil(n_weekend / EST_STATIONS_PER_CHUNK))

    any_per_type = math.ceil(n_any / 2) if n_any > 0 else 0
    wd_chunks_needed = max(wd_chunks_needed,
                           math.ceil((n_weekday + any_per_type) / EST_STATIONS_PER_CHUNK))
    we_chunks_needed = max(we_chunks_needed,
                           math.ceil((n_weekend + any_per_type) / EST_STATIONS_PER_CHUNK))

    # At least one chunk per day
    wd_chunks_needed = max(wd_chunks_needed, len(weekday_days))
    we_chunks_needed = max(we_chunks_needed, len(weekend_days))

    def pick_starts(n, rng):
        picks = []
        pool = list(valid_starts)
        rng.shuffle(pool)
        while len(picks) < n:
            if not pool:
                pool = list(valid_starts)
                rng.shuffle(pool)
            picks.append(pool.pop())
        return picks

    wd_starts = pick_starts(wd_chunks_needed, rng)
    we_starts = pick_starts(we_chunks_needed, rng)

    def spread_across_days(n_chunks, days):
        return [days[i % len(days)] for i in range(n_chunks)]

    wd_days = spread_across_days(wd_chunks_needed, weekday_days)
    we_days = spread_across_days(we_chunks_needed, weekend_days)

    chunks = []
    for (h, m, label), d in zip(wd_starts, wd_days):
        chunks.append({"day": d, "start_h": h, "start_m": m,
                       "start_label": label, "is_weekend": False})
    for (h, m, label), d in zip(we_starts, we_days):
        chunks.append({"day": d, "start_h": h, "start_m": m,
                       "start_label": label, "is_weekend": True})

    # De-conflict same-day chunks
    chunks.sort(key=lambda c: (c["day"], c["start_h"], c["start_m"]))
    for i in range(1, len(chunks)):
        prev, curr = chunks[i - 1], chunks[i]
        if curr["day"] == prev["day"]:
            prev_start = prev["start_h"] * 60 + prev["start_m"]
            curr_start = curr["start_h"] * 60 + curr["start_m"]
            min_start = prev_start + chunk_min
            if curr_start < min_start:
                new_start = min(min_start, HARD_END - chunk_min)
                if new_start > curr_start:
                    curr["start_h"] = new_start // 60
                    curr["start_m"] = new_start % 60

    chunks.sort(key=lambda c: (c["day"], c["start_h"], c["start_m"]))
    return chunks


def build_chunk_route(visits, chunk_budget_min, travel_fn, rng):
    if not visits:
        return [], []

    remaining = list(visits)
    routed = []
    elapsed = 0

    first = rng.choice(remaining)
    remaining.remove(first)
    first.travel_from = "start"
    first.travel_min = 0
    first.travel_mode = "start"
    elapsed += OBSERVATION_MIN
    routed.append(first)

    while remaining:
        current = routed[-1].station
        candidates = []
        for v in remaining:
            t, mode = travel_fn(current, v.station)
            if elapsed + t + OBSERVATION_MIN <= chunk_budget_min:
                candidates.append((t, mode, v))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        t, mode, best = candidates[0]
        remaining.remove(best)
        best.travel_from = current.name
        best.travel_min = t
        best.travel_mode = mode
        elapsed += t + OBSERVATION_MIN
        routed.append(best)

    return routed, remaining


def schedule(visit_pool, chunks, chunk_budget_min, travel_fn, rng):
    wd_visits = [v for v in visit_pool if v.day_type_req == "weekday"]
    we_visits = [v for v in visit_pool if v.day_type_req == "weekend"]
    any_visits = [v for v in visit_pool if v.day_type_req == "any"]

    rng.shuffle(wd_visits)
    rng.shuffle(we_visits)
    rng.shuffle(any_visits)

    wd_chunks = [c for c in chunks if not c["is_weekend"]]
    we_chunks = [c for c in chunks if c["is_weekend"]]

    chunk_assignments = {id(c): [] for c in chunks}

    for i, v in enumerate(wd_visits):
        chunk_assignments[id(wd_chunks[i % len(wd_chunks)])].append(v)
    for i, v in enumerate(we_visits):
        chunk_assignments[id(we_chunks[i % len(we_chunks)])].append(v)

    all_chunks_ordered = wd_chunks + we_chunks
    for v in any_visits:
        best_c = min(all_chunks_ordered, key=lambda c: len(chunk_assignments[id(c)]))
        chunk_assignments[id(best_c)].append(v)

    all_routes = []
    for ci, chunk in enumerate(chunks):
        assigned = chunk_assignments[id(chunk)]
        routed, _ = build_chunk_route(assigned, chunk_budget_min, travel_fn, rng)

        start_min = chunk["start_h"] * 60 + chunk["start_m"]
        clock = start_min
        for v in routed:
            v.chunk_idx = ci + 1
            v.day = chunk["day"] + 1
            v.day_type = "weekend" if chunk["is_weekend"] else "weekday"
            clock += v.travel_min
            v.arrive_time = minutes_to_hhmm(clock)
            clock += OBSERVATION_MIN
            v.depart_time = minutes_to_hhmm(clock)

        all_routes.append((chunk, routed))

    scheduled_ids = {id(v) for _, route in all_routes for v in route}
    unassigned = [v for v in visit_pool if id(v) not in scheduled_ids]

    return all_routes, unassigned


# ---------------------------------------------------------------------------
# 6. OUTPUT
# ---------------------------------------------------------------------------

def print_plan(all_routes, unassigned, sampled_stations, frame, seed, args):
    n_sampled = len(sampled_stations)
    total_visits = sum(len(route) for _, route in all_routes)
    total_obs = total_visits * OBSERVATION_MIN
    total_travel = sum(v.travel_min for _, route in all_routes for v in route)
    total_field = total_obs + total_travel
    n_chunks = len(all_routes)
    budget_min = int(args.budget_hours * 60)

    print("=" * 80)
    print("METRO STATION SAMPLING PLAN")
    print(f"Seed: {seed}  |  Sampled: {n_sampled} of {len(frame)} stations  |  "
          f"Days: {args.n_days}  |  Visits/station: {args.visits_per_station}")
    print(f"Frame: {args.frame}  |  "
          f"Budget: {args.budget_hours}h in ~{args.chunk_hours}h chunks")
    print(f"Observation per visit: {OBSERVATION_MIN} min "
          f"(gate {PROTOCOL['point_a']['duration_min']} + "
          f"platform {PROTOCOL['point_b']['duration_min']} + "
          f"transition {PROTOCOL['setup_transition_min']})")
    print(f"Scheduled: {total_visits} visits in {n_chunks} chunks  |  "
          f"Observation: {total_obs} min  |  Travel: {total_travel} min  |  "
          f"Total field time: {total_field // 60}h {total_field % 60}m")
    if unassigned:
        print(f"WARNING: {len(unassigned)} visits could not be scheduled")
    if total_field > budget_min:
        print(f"WARNING: Field time ({total_field} min) exceeds budget ({budget_min} min)")
    print("=" * 80)

    # Sampled stations
    print("\nSAMPLED STATIONS")
    print("-" * 80)
    print(f"{'Station':<28} {'Line':<8} {'Area':<30} {'Lat':>8} {'Lon':>8}")
    print("-" * 80)
    for s in sorted(sampled_stations, key=lambda s: (s.line, s.name)):
        ix = " *" if s.interchange else ""
        print(f"{s.name + ix:<28} {s.line:<8} {s.area:<30} {s.lat:>8.4f} {s.lon:>8.4f}")
    print("* = interchange station")

    # Line coverage
    line_sampled = Counter(s.line for s in sampled_stations)
    line_frame = Counter(s["line"] for s in frame)
    print(f"\nLINE COVERAGE")
    print(f"{'Line':<10} {'Frame':<8} {'Sampled':<8}")
    for line in sorted(line_frame):
        print(f"{line:<10} {line_frame[line]:<8} {line_sampled.get(line, 0):<8}")

    # Chunk itineraries
    for ci, (chunk, route) in enumerate(all_routes):
        day_type = "Weekend" if chunk["is_weekend"] else "Weekday"
        start_hhmm = f"{chunk['start_h']:02d}:{chunk['start_m']:02d}"
        chunk_travel = sum(v.travel_min for v in route)
        chunk_obs = len(route) * OBSERVATION_MIN
        chunk_total = chunk_travel + chunk_obs

        print(f"\n{'=' * 80}")
        print(f"CHUNK {ci + 1}: Day {chunk['day'] + 1} ({day_type}) | "
              f"Start {start_hhmm} | {len(route)} stations | "
              f"{chunk_total // 60}h {chunk_total % 60}m field time")
        print(f"{'=' * 80}")

        for v in route:
            if v.travel_mode == "start":
                print(f"  {v.arrive_time}  Arrive {v.station.name}")
            else:
                print(f"         -> {v.station.name} "
                      f"({v.travel_mode} {v.travel_min} min from {v.travel_from})")
                print(f"  {v.arrive_time}  Arrive {v.station.name}")
            print(f"         Observe: gate {PROTOCOL['point_a']['duration_min']} min + "
                  f"platform {PROTOCOL['point_b']['duration_min']} min")
            print(f"  {v.depart_time}  Done")

        if route:
            print(f"\n  Observation: {chunk_obs} min | Travel: {chunk_travel} min | "
                  f"Total: {chunk_total} min")

    # Visit matrix
    print(f"\n{'=' * 80}")
    print("VISIT MATRIX")
    print(f"{'=' * 80}")
    col_w = 20
    header = f"{'Station':<28}" + "".join(
        f"{'Day ' + str(d+1):<{col_w}}" for d in range(args.n_days))
    print(header)
    print("-" * (28 + col_w * args.n_days))
    for s in sorted(sampled_stations, key=lambda s: s.name):
        row = f"{s.name:<28}"
        for d in range(args.n_days):
            visits_on_day = [v for _, route in all_routes for v in route
                            if v.station.name == s.name and v.day == d + 1]
            if visits_on_day:
                times = " ".join(v.arrive_time for v in visits_on_day)
                row += f"{times:<{col_w}}"
            else:
                row += f"{'--':<{col_w}}"
        print(row)


def write_csv(all_routes, seed, filepath):
    rows = []
    for chunk, route in all_routes:
        for v in route:
            rows.append({
                "seed": seed,
                "chunk": v.chunk_idx,
                "day": v.day,
                "day_type": v.day_type,
                "station": v.station.name,
                "line": v.station.line,
                "lat": v.station.lat,
                "lon": v.station.lon,
                "area": v.station.area,
                "interchange": v.station.interchange,
                "arrive": v.arrive_time,
                "depart": v.depart_time,
                "observation_min": OBSERVATION_MIN,
                "travel_from": v.travel_from,
                "travel_min": v.travel_min,
                "travel_mode": v.travel_mode,
            })

    rows.sort(key=lambda r: (r["day"], r["arrive"]))

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to {filepath} ({len(rows)} visit rows)")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Metro station sampling plan generator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame", type=str, default="station_frame.csv",
                        help="Path to station frame CSV")
    parser.add_argument("--n_stations", type=int, default=9,
                        help="Number of stations to sample")
    parser.add_argument("--n_days", type=int, default=4)
    parser.add_argument("--visits_per_station", type=int, default=2)
    parser.add_argument("--budget_hours", type=float, default=16)
    parser.add_argument("--chunk_hours", type=float, default=4)
    parser.add_argument("--force_include", type=str, nargs="*", default=["Majestic"])
    parser.add_argument("--csv", type=str, default="metro_sampling_plan.csv")
    args = parser.parse_args()

    # Load frame
    frame_path = Path(args.frame)
    if not frame_path.exists():
        print(f"ERROR: Station frame not found at {frame_path}")
        print("Expected CSV with columns: name, line, lat, lon, area, interchange")
        return
    frame = load_frame(frame_path)
    print(f"Loaded {len(frame)} stations from {frame_path}")

    rng = random.Random(args.seed)

    # Sample
    sampled = sample_stations(
        frame, args.n_stations, rng,
        force_include=set(args.force_include) if args.force_include else None,
    )

    # Visits
    visit_pool = create_visit_pool(sampled, args.visits_per_station)

    # Chunks
    chunks = define_chunks(visit_pool, args.n_days, rng,
                          budget_hours=args.budget_hours,
                          chunk_hours=args.chunk_hours)

    # Schedule
    chunk_budget_min = int(args.chunk_hours * 60)
    all_routes, unassigned = schedule(visit_pool, chunks, chunk_budget_min, travel_time, rng)

    # Output
    print_plan(all_routes, unassigned, sampled, frame, args.seed, args)
    write_csv(all_routes, args.seed, filepath=args.csv)


if __name__ == "__main__":
    main()
