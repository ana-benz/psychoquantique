import streamlit as st
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


st.title("Simulateur d'États Quantiques")

# Paramètres globaux
g0 = st.slider("A quel point votre humeur fluctue ?", 0.01, 1.0, 0.3)
gamma_ug = st.slider("Votre tendance à l'euphorie ?", 0.001, 0.1, 0.01)
gamma_dg = st.slider("Votre tendance à la dépression ?", 0.001, 0.1, 0.01)
gamma_gg = st.slider("Votre tendance à ne pas rester neutre ?", 0.0001, 0.001, 0.0005)

# Échelle de temps en jours
t_max = st.slider("Durée de la simulation (jours)", 10, 365, 30)
t_points = 1000
t_list = np.linspace(0, t_max, t_points)

st.subheader("Définition manuelle des triggers")
num_triggers = st.number_input("Nombre de triggers", min_value=0, max_value=5, value=3)

trigger_times = []
trigger_effects = []

for i in range(num_triggers):
    col1, col2 = st.columns(2)
    with col1:
        t_trig = st.slider(f"Temps du trigger {i+1} (jour)", 0, t_max, int(t_max/(num_triggers+1))*(i+1))
    with col2:
        effect = st.slider(f"Intensité du trigger {i+1}", 0.1, 2.0, 1.0)
    trigger_times.append(t_trig)
    trigger_effects.append(effect)

st.subheader("État initial")
# Curseur pour choisir l'état initial :
# Valeur -1 -> pure dépression, 0 -> neutre, 1 -> pure euphorie.
initial_state_val = st.slider("Choix de l'état initial (Dépression [-1] – Neutre [0] – Euphorie [1])",
                              -1.0, 1.0, 0.0, step=0.01)
# Définition des coefficients (la somme vaut toujours 1)
w_dep = max(0, -initial_state_val) # Poids pour dépression
w_neu = 1 - abs(initial_state_val) # Poids pour état neutre
w_euph = max(0, initial_state_val) # Poids pour euphorie

st.write(f"Coefficients: Dépression = {w_dep:.2f}, Neutre = {w_neu:.2f}, Euphorie = {w_euph:.2f}")

if st.button("Lancer la simulation"):
    

    # États quantiques : |g> (neutre), |u> (euphorique), |d> (dépressif)
    g = qt.basis(3, 0)
    u = qt.basis(3, 1)
    d = qt.basis(3, 2)

    # Opérateurs de projection
    P_g = g * g.dag()
    P_u = u * u.dag()
    P_d = d * d.dag()

    # Opérateurs de transition
    sigma_gu = u * g.dag()
    sigma_ug = g * u.dag()
    sigma_gd = d * g.dag()
    sigma_dg = g * d.dag()
    sigma_ud = d * u.dag()
    sigma_du = u * d.dag()
    sigma_z = qt.Qobj([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    resilience_factor = 0.9

    # Fonctions de couplage dépendant du temps
    def Omega_gu(t, args):
        effet = sum(
            (resilience_factor ** i) * (np.cos(effect / 2 * t)) * np.exp(-2 * effect * (t - t_trig) ** 2)
            for i, (t_trig, effect) in enumerate(zip(trigger_times, trigger_effects)) if t >= t_trig
        )
        return 0.1 - effet

    def Omega_gd(t, args):
        effet = sum(
            (resilience_factor ** i) * (np.sin(effect / 2 * t)) * np.exp(-2 * effect * (t - t_trig) ** 2)
            for i, (t_trig, effect) in enumerate(zip(trigger_times, trigger_effects)) if t >= t_trig
        )
        return 0.1 - effet

    def Omega_ud(t, args):
        effet = sum(
            (resilience_factor ** i) * np.sqrt(effect) * np.exp(-2 * effect * (t - t_trig) ** 2)
            for i, (t_trig, effect) in enumerate(zip(trigger_times, trigger_effects)) if t >= t_trig
        )
        return 0.05 - effet

    # Hamiltonien dépendant du temps
    H = [[sigma_gu + sigma_ug, Omega_gu],
         [sigma_gd + sigma_dg, Omega_gd],
         [sigma_ud + sigma_du, Omega_ud],
         g0 * sigma_z]

    # Opérateurs de saut pour la dissipation
    c_ops = [
        np.sqrt(gamma_ug) * sigma_ug, # u → g
        np.sqrt(gamma_dg) * sigma_dg, # d → g
        np.sqrt(gamma_gg) * (sigma_gd + sigma_gu) # g → (u ou d)
    ]

    # État initial : combinaison linéaire normalisée
    psi0 = w_dep * d + w_neu * g + w_euph * u

    # Simulation
    result = qt.mesolve(H, psi0, t_list, c_ops, [P_g, P_u, P_d])
    p_g, p_u, p_d = result.expect

    # Tracé des résultats
    fig, ax = plt.subplots()
    for t_trig, effect in zip(trigger_times, trigger_effects):
        ax.axvline(x=t_trig, color='orange', linestyle='dashed', lw=4 * effect,
                   label="Trigger" if t_trig == trigger_times[0] else "")
    ax.plot(t_list, p_g, label="Neutre (g)", color="blue")
    ax.plot(t_list, p_u, label="Euphorie (u)", color="red")
    ax.plot(t_list, p_d, label="Dépression (d)", color="green")
    ax.set_xlabel("Temps (jours)")
    ax.set_ylabel("Probabilité d'occupation")
    ax.set_title("Évolution des états Psychoquantiques")
    ax.legend()

    st.pyplot(fig)
    st.success("Simulation terminée :) !")
