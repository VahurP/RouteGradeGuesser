import json

class Climb:
    def __init__(self, name, grade, route):
        self.name   = name
        self.grade  = grade
        self.route  = route  # list of moves

    def __repr__(self):
        return f"<Climb {self.name!r}, grade={self.grade!r}, moves={len(self.route)}>"

# adjust to your path style
PATH_TO_JSON = r"C:\Users\vahurpaist\Downloads\problems_2023_01_30\problems MoonBoard 2016 .json"

# 1. load the file
with open(PATH_TO_JSON, encoding="utf-8") as f:
    payload = json.load(f)

# 2. extract the list of climbs
raw_climbs = payload["data"]      # a list of dicts

# 3. instantiate your Climb class for each entry
climbs = [
    Climb(item["name"], item["grade"], item["moves"])
    for item in raw_climbs
]

# now `climbs` is a list of Climb objects

for climb in climbs:
    print(f"{climb.name}  ({climb.grade}):")
    # climb.route is a list of move‑dicts
    labels = [move["description"] for move in climb.route]
    print("  →", " →".join(labels))
    print()