import json

class Route:
    def __init__(self, name, grade, route):
        self.name = name
        self.grade = grade
        self.route = route

    def __repr__(self):
        return f"<Route {self.name!r}, grade={self.grade!r}, moves={len(self.route)}>"


PATH_TO_JSON = r"C:\Users\vahurpaist\Downloads\problems_2023_01_30\problems MoonBoard 2016 .json"

with open(PATH_TO_JSON, encoding="utf-8") as f:
    payload = json.load(f)

raw_climbs = payload["data"]

climbs = [
    Route(item["name"], item["grade"], item["moves"])
    for item in raw_climbs
]

for climb in climbs:
    print(f"{climb.name} ({climb.grade}):")
    labels = [move["description"] for move in climb.route]
    print("  →", " →".join(labels))
    print()
