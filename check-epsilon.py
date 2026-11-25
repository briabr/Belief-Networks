import pandas as pd

for eps in [0.0, 1.0]:

    path = (
        f"simOut_n1_nopressure/"
        f"sim_init_w0.20_beta3.00_eps{eps:.2f}_seed0.csv"
    )

    df = pd.read_csv(path)

    print(f"\nEPSILON = {eps}")

    print(
        df[[
            "t",
            "absOm_tot",
            "absOm_foc",
            "expI",
            "x_focal",
            "extr_nonfoc"
        ]].to_string(index=False)
    )
