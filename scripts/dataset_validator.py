"""Print the fields present in an episode; no audit artifacts."""
import argparse,json
from imitation_lab.raw import Episode,describe
p=argparse.ArgumentParser();p.add_argument('episode');a=p.parse_args()
with Episode(a.episode) as ep:print(json.dumps(describe(ep),ensure_ascii=False,indent=2))
