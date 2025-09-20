# test_maps_free.py
import requests, sys, time
from urllib.parse import quote_plus

USER_AGENT = "CareSetu/1.0 (your-email@example.com)"  # change to your project/email per Nominatim policy

def nominatim_geocode(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return {
        "display_name": data[0].get("display_name"),
        "lat": float(data[0]["lat"]),
        "lon": float(data[0]["lon"]),
        "raw": data[0]
    }

def overpass_nearby_hospitals(lat, lon, radius=3000):
    # Query nodes/ways/relations with amenity=hospital or healthcare=clinic
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})[amenity=hospital];
      way(around:{radius},{lat},{lon})[amenity=hospital];
      relation(around:{radius},{lat},{lon})[amenity=hospital];
      node(around:{radius},{lat},{lon})[healthcare=clinic];
      way(around:{radius},{lat},{lon})[healthcare=clinic];
      relation(around:{radius},{lat},{lon})[healthcare=clinic];
    );
    out center 20;
    """
    url = "https://overpass-api.de/api/interpreter"
    headers = {"User-Agent": USER_AGENT}
    r = requests.post(url, data={"data": query}, headers=headers, timeout=30)
    r.raise_for_status()
    res = r.json()
    results = []
    for el in res.get("elements", []):
        # get a name if available and coordinates
        name = el.get("tags", {}).get("name") or el.get("tags", {}).get("operator") or "Unknown"
        lat_e = el.get("lat") or (el.get("center") or {}).get("lat")
        lon_e = el.get("lon") or (el.get("center") or {}).get("lon")
        kind = el.get("type")
        addr = ", ".join([v for k, v in el.get("tags", {}).items() if k.startswith("addr:")]) or None
        results.append({
            "name": name,
            "lat": lat_e,
            "lon": lon_e,
            "tags": el.get("tags", {}),
            "addr": addr,
            "type": kind
        })
    return results

if __name__ == "__main__":
    q = "208001 Kanpur"   # replace with PIN or "Kanpur, Uttar Pradesh" etc
    print("Geocoding:", q)
    g = nominatim_geocode(q)
    if not g:
        print("No geocode result. Try a different query.")
        sys.exit(1)
    print("Location:", g["display_name"], g["lat"], g["lon"])
    # polite pause
    time.sleep(1)
    hospitals = overpass_nearby_hospitals(g["lat"], g["lon"], radius=3000)
    print("Found hospitals:", len(hospitals))
    for i, h in enumerate(hospitals[:10], start=1):
        print(f"{i}. {h['name']} — {h['addr']} — {h['lat']},{h['lon']}")
