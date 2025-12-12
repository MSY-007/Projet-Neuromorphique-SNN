'''
1. Prédiction simple du vent pour les prochaines heures (moyenne glissante)
2. Téléchargement des données en CSV (vent, spikes, sorties neurones)
3. Interface claire avec résumé et graphique
'''
import streamlit as st
import requests
import torch
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from norse.torch.functional.lif import lif_step, LIFState, LIFParameters

#--- Configuration ---
st.set_page_config(page_title="Neurométéo Afrique de l'Ouest", layout="wide")
st.title("🧠 Simulation Neuromorphique des Vents")

#--- Sélection de la ville ---
villes = {
    "Abidjan": (5.34, -4.03),
    "Korhogo": (9.46, -5.63),
    "Yamoussoukro": (6.82, -5.28),
    "Bouaké": (7.69, -5.03),
    "San Pedro": (4.75, -6.65)
}
ville_nom = st.selectbox("📍 Ville", list(villes.keys()))
lat, lon = villes[ville_nom]

#--- Récupération des données météo --
try:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m"
    data = requests.get(url).json()
    wind = data["hourly"]["wind_speed_10m"][:24]
    wind_norm = [(w - min(wind)) / (max(wind) - min(wind)) for w in wind]
    threshold = st.slider("🎯 Seuil d'activation du neurone", 0.1, 1.0, 0.3, 0.05)
    spikes = [1 if w > threshold else 0 for w in wind_norm]
except Exception as e:
    st.error(f"Erreur récupération météo : {e}")
    st.stop()

#--- Simulation LIF ---
state = LIFState(z=torch.zeros(1), v=torch.zeros(1), i=torch.zeros(1))
w_in = torch.eye(1)
w_rec = torch.eye(1)
outputs = []
for spike in spikes:
    inp = torch.tensor([float(spike)])
    z, state = lif_step(inp, state, w_in, w_rec, p=LIFParameters(alpha=100.0))
    outputs.append(z.item())

#--- Prédiction simple : moyenne glissante ---
pred_window = st.slider("⏳ Fenêtre de prédiction (heures)", 1, 5, 3)
preds = [sum(wind[i:i+pred_window]) / pred_window for i in range(len(wind)-pred_window+1)]
preds += [None] * (24 - len(preds))  # pour aligner

#--- Affichage graphique ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wind, label="🌬️ Vent réel (km/h)", color="blue")
ax.plot(wind_norm, label="Vent normalisé", color="skyblue")
ax.plot(spikes, label="⚡ Spikes météo", linestyle="--", color="orange")
ax.plot(outputs, label="🧠 Sortie neurone", linestyle=":", color="green")
ax.plot(preds, label="📈 Prédiction vent", linestyle="-.", color="purple")
ax.axhline(y=threshold, color='red', linestyle='--', label="🔴 Seuil critique (normalisé)")
ax.set_title(f"Réaction neuromorphique - {ville_nom}")
ax.set_xlabel("Heure")
ax.set_ylabel("Valeurs")
ax.grid(True)
ax.legend()
st.pyplot(fig)

#--- Résumé ---
st.markdown("### 🔎 Analyse")
st.write(f"- Spikes détectés : {sum(spikes)} / 24")
st.write(f"- Max sortie neurone : {max(outputs):.2f}")
st.write(f"- Max vent (km/h) : {max(wind):.1f}")

val = preds[len(spikes)-1] if preds and len(preds) >= len(spikes) else None

if isinstance(val, (float, int)):
    st.write(f"- Vent prédit (prochaine heure) : {val:.2f} km/h")
else:
    st.warning("Prévision indisponible ou non valide.")



#--- Export CSV ---
df = pd.DataFrame({
    "Heure": list(range(24)),
    "Vent (km/h)": wind,
    "Vent normalisé": wind_norm,
    "Spike": spikes,
    "Sortie neurone": outputs,
    "Vent prédit (moy glissante)": preds
})
csv = df.to_csv(index=False)
st.download_button("⬇️ Télécharger les données CSV", data=csv, file_name="neurometeo.csv", mime="text/csv")
#--- Alerte e-mail ---
alerte_active = st.checkbox("📧 Activer alerte e-mail si vent > 30 km/h")
if alerte_active and max(wind) > 30:
    try:
        sender = "ton.email@gmail.com"
        password = "motdepasse_application"  # ⚠️ mot de passe d'application Gmail
        receiver = st.text_input("✉️ E-mail de destination", "destinataire@example.com")

        if st.button("Envoyer alerte maintenant"):
            msg = MIMEText(f"Alerte : Vent fort détecté à {ville_nom} ({max(wind):.1f} km/h)")
            msg["Subject"] = "🌪️ Alerte Vent Fort"
            msg["From"] = sender
            msg["To"] = receiver

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)

            st.success("✅ Alerte envoyée avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de l'envoi : {e}")