
from sigma_redshift_toy_models import generate_synthetic_sne, fit_all_models_to_sn, print_model_rankings, plot_hubble_comparison, save_results_json

if __name__ == "__main__":
    data = generate_synthetic_sne(n=420, zmin=0.01, zmax=2.0, H0=70.0, Om=0.3, Ol=0.7, sigma_int=0.12, seed=42)
    results = fit_all_models_to_sn(data["z"], data["mu"], data["sigma_mu"])
    print_model_rankings(results)
    fig_path = plot_hubble_comparison(data["z"], data["mu"], data["sigma_mu"], results, filename="hubble_toy_results.png")
    json_path = save_results_json(results, path="toy_fit_results.json")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved fit results to: {json_path}")
