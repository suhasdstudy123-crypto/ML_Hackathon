"""Adapter around inspected route-blind notebook definitions; no approximation."""
from backend.app.pipeline.notebook_features import TripData, build_trip_table
from backend.app.core.errors import AppError

class NotebookFeatureService:
    def __init__(self, compatibility, demo_mode=False):
        self.compatibility, self.demo_mode = compatibility, demo_mode

    def generate(self, trip, check_deadline=lambda: None):
        policy = self.compatibility.require_generation_ready(self.demo_mode)
        check_deadline()
        if len(trip.arcs) < 5:
            raise AppError('INSUFFICIENT_CANDIDATES', 'Notebook contract requires at least five directed road arcs')
        # This type has no .route field and receives no evaluation service/reference.
        source = TripData(trip.trip_id, trip.nodes[['lon','lat']].to_numpy(),
                          trip.arcs[['u','v']].to_numpy(), trip.track[['lon','lat','time']].to_numpy())
        data = build_trip_table(source)
        check_deadline()
        data.attrs['pipeline_warnings'] = policy.get('warnings', [])
        return data
