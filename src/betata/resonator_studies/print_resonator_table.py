""" """

from betata.resonator_studies.resonator import load_resonators

if __name__ == "__main__":
    """ """

    resonators = load_resonators()
    resonator_dict = {}
    for resonator in resonators:
        film_idx = int(resonator.name.split("_")[1][1:])
        film_token = f"F{film_idx:02d}"
        rr_type = resonator.type
        rr_pitch = f"{resonator.pitch * 1e6:03.0f}"
        rr_label = f"{film_token}--{rr_type}--{rr_pitch}"
        resonator_dict[rr_label] = resonator

    sorted_resonator_dict = dict(sorted(resonator_dict.items()))

    for resonator_name, resonator in sorted_resonator_dict.items():
        p_ms_token = f"{resonator.p_ms * 1e4:.2f}"
        q_c_token = f"{resonator.Q_c * 1e-6:.2f}"
        fr_token = f"{resonator.fr_bare * 1e-9:.3f}"
        alpha_token = f"{resonator.alpha_bare:.3f}"
        if resonator.l_sheet is None:
            lk_token = "---"
        else:
            lk_token = (
                f"{resonator.l_sheet * 1e12:.1f} $\pm$ {resonator.l_sheet_err * 1e12:.1f}"
            )

        if resonator.qpt_fit_params is not None:
            qtls0_token = f"{resonator.qpt_fit_params['Q_TLS0']['value'] * 1e-6:.3f} $\pm$ {resonator.qpt_fit_params['Q_TLS0']['stderr'] * 1e-6:.3f}"
            qother_token = f"{resonator.qpt_fit_params['Q_other']['value'] * 1e-6:.3f} $\pm$ {resonator.qpt_fit_params['Q_other']['stderr'] * 1e-6:.3f}"
        else:
            qtls0_token = "---"
            qother_token = "---"

        latex_token = f"{resonator_name} & {p_ms_token} & {q_c_token} & {fr_token} & {alpha_token} & {lk_token} & {qtls0_token} & {qother_token} \\\\"

        print(latex_token)
