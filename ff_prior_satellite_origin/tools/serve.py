"""Read-only loopback media server; no physics entry point."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ff_prior_balanced_delivery.tools import serve
serve.PORT=8772
serve.ROUTES={'/':ROOT/'ff_prior_satellite_origin/delivery_001','/previous/':ROOT/'ff_prior_balanced_delivery/final_brake_001'}
if __name__=='__main__':serve.main()
