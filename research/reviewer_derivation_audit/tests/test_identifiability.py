from sigma_sprint.identifiability import dependency_audit


def test_expected_jacobian_ranks():
    audit = dependency_audit()
    assert audit["deep_btfr"]["rank"] == 1
    assert audit["deep_btfr"]["nullity"] == 1
    assert audit["fox_clusters_fixed_L"]["rank"] == 1
    assert audit["fox_clusters_fixed_L"]["nullity"] == 2
    assert audit["hypothetical_varied_L"]["rank"] == 2
    assert audit["cluster_coherence_fixed_zero"]["rank"] == 1
