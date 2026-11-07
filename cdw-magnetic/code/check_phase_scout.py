"""Audit the scout map, spectral reuse, old witnesses and selected N=512 checks."""

import json
import math
from pathlib import Path

import numpy as np

from adaptive_search import json_finite
from basis_bounds import relative_interval
from homogeneous_potential import minimize_mass, thermal_weight
from phase_scout import spectral_bank
from refine_cdw_candidate import CorrectedObjective, validate_point
from relative_potential import RelativeBlock


if __name__=="__main__":
    report=json.loads(Path("cdw-magnetic/code/phase-scout.json").read_text())
    assert report["completed"]
    points={(p["B"],p["mu"]):p for row in report["rows"] for p in row["points"]}
    assert len(points)==len(report["field_axis"])*len(report["chemical_axis"])==500
    for p in points.values():
        if p["status"]!="confirmed_cdw":continue
        assert len(p["validation"])>=2
        valid=p["validation"]
        assert all(v["lower"]<=v["upper"] for v in valid)
        assert valid[-1]["upper"] < -1e-8
        assert max(v["lower"] for v in valid)<=min(v["upper"] for v in valid)
        assert all(v["direct_frequency_discrepancy"]<=v["frequency_error_bound"]+5e-10 for v in valid)
        assert abs(p["refined_reference"]["potential"]-p["homogeneous_reference"]["potential"])<1e-8
    previous=json.loads(Path("cdw-magnetic/code/cdw-branch-continuation.json").read_text())
    witness_count=0
    for name in ("mu_cut","field_cut"):
        for p in previous[name]:
            key=p["B"],p["mu"]
            if p["status"]=="confirmed_improving_point" and key in points:
                assert points[key]["status"]=="confirmed_cdw",key
                witness_count+=1
    # Check shared-bank thermal reweighting against the previous independent
    # per-mu implementation, including far-from-witness backgrounds.
    B,T=.2,.02
    coordinates,E,bases,failures=spectral_bank(B,T)
    assert not failures
    reuse=[]
    for M,b in ((.025,.15),(.2,.8),(1.1,.025)):
        index=np.flatnonzero(np.all(np.isclose(coordinates,(M,b),atol=1e-14),axis=1))[0]
        for mu in (.8,1.05,1.4):
            direct=CorrectedObjective(B,mu,T,64,0)((M,b))
            reused=bases[index]-B/(2*math.pi)*np.sum(thermal_weight(E[index],mu,T))
            assert abs(reused-direct)<2e-10
            reuse.append({"mass":M,"wave":b,"mu":mu,"difference":float(reused-direct)})
    confirmed=[p for p in points.values() if p["status"]=="confirmed_cdw"]
    selected=[min(confirmed,key=lambda p:p["validation"][-1]["upper"]),
              max(confirmed,key=lambda p:p["validation"][-1]["upper"]),
              points[(.2,1.05)],points[(.3,1.125)]]
    audit={"representative_checks":[],"unresolved_retries":[],"spectral_reuse_checks":reuse,
           "previous_witness_matches_including_duplicate_center":witness_count}
    for p in selected:
        B,mu=p["B"],p["mu"]
        reference=minimize_mass(B,mu,T,grid_points=401,mass_max=8)
        assert abs(reference.potential-p["refined_reference"]["potential"])<2e-9
        point=p["selected_point"]
        result=validate_point(B,mu,T,(point["mass"],point["wave"]),reference.potential,cutoffs=(512,))
        assert result[-1]["gap_upper"]<0
        audit["representative_checks"].append({"B":B,"mu":mu,"reference_401":reference.potential,
                                                "validation":result})
    for p in points.values():
        if p["status"]!="unresolved":continue
        point=p["selected_point"]
        retries=[]
        for N in (256,512):
            try:
                block=RelativeBlock(p["B"],point["mass"],point["wave"],N)
                interval=relative_interval(block,p["mu"],T,frequency_tolerance=1e-9)
                retries.append({"N":N,"lower_relative":interval.lower,"upper_relative":interval.upper,
                                "vacuum_upper_available":interval.vacuum.supported})
            except (ArithmeticError,FloatingPointError) as error:
                retries.append({"N":N,"failure":str(error)})
        audit["unresolved_retries"].append({"B":p["B"],"mu":p["mu"],
                                           "actual_checked_point":point,"retries":retries})
    Path("cdw-magnetic/code/phase-scout-audit.json").write_text(
        json.dumps(json_finite(audit),indent=2,allow_nan=False)+"\n")
    print("PASS scout: map invariants, previous witnesses, independent spectral reuse, "
          "representative N=512 checks, dense homogeneous references. Unresolved retries recorded.")
