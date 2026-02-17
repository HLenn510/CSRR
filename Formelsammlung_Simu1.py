import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Daten importieren ---
# Wichtig: Trennzeichen (sep) und Dezimaltrennzeichen (decimal) beachten
df = pd.read_csv("Temp_loads_pv_hourly_utf_8.csv", sep=";", decimal=",")

# --- Netzwerk initialisieren ---
n = pypsa.Network()

# --- Zeitachse setzen (Snapshots) ---
# Übernimmt den Index der CSV als Zeitbasis für das Netz
n.set_snapshots(df.index)

# --- Busse (Knotenpunkte) definieren ---
# Standard Strom-Bus
n.add("Bus", name="electricity")
# Sektorkopplung (Wärme, Wasser, Diesel)
n.add("Bus", name="thermal")
n.add("Bus", name="hot_water")
n.add("Bus", name="diesel")



# --- A. Einheiten Umrechnungen ---
# Watt zu kW
leistung_kw = 2000 / 1000 
# Diesel: Energiegehalt (kWh pro Liter)
diesel_energy_density = 9.8  # kWh/L
# Warmwasser: Speicherkapazität (Liter -> kWh)
# Formel: (Liter * Wärmekapazität_Wasser * Delta_T) / 3600
# 4.18 kJ/kg*K ist die Wärmekapazität, 40K ist die Temperaturdifferenz
cap_kwh = (30 * 4.18 * 40) / 3600 


# --- B. Temperaturabhängige Effizienz (Interpolation) ---
# Das Herzstück deiner Wärmepumpen-Logik
# np.interp(x-wert-aktuell, x-werte-kennlinie, y-werte-kennlinie)

outside_temp = df["Außentemperatur [ºC]"]

# Stützstellen (Kennlinie aus Datenblatt)
temp_points = [-20, -15, -7, 7, 20, 35]
cop_points =  [1.82, 2.06, 2.52, 3.67, 5.65, 8.70]

# Berechnung der Zeitreihe für das ganze Jahr
cop_time_series = np.interp(outside_temp, temp_points, cop_points)

# Berechnung der maximalen Leistung (p_max_pu) basierend auf Temperatur
# Leistung_aktuell / Nennleistung
el_power_points = [1.33, 1.39, 1.46, 1.02, 0.98, 0.88]
p_nom_ref = 0.8
p_max_pu_series = np.interp(outside_temp, temp_points, el_power_points) / p_nom_ref


# --- A. Die Annuitäten-Formel (PMT) ---
# Verwandelt Einmalkosten (Invest) in jährliche Kosten
# r = Zinssatz (3%), t = Lebensdauer in Jahren
def annuity(invest, t, r=0.03):
    return invest * (r * (1 + r) ** t) / ((1 + r) ** t - 1)

# Anwendung:
capex_total = 60000
lifetime = 10
annual_cost = annuity(capex_total, lifetime)


# --- B. Grenzkosten (Marginal Cost) ---
# Kosten pro verbrauchter Einheit (z.B. Dieselgenerator)
diesel_price = 1.70 # Euro/Liter
# Kosten pro kWh = Preis pro Liter / Energie pro Liter
mc_diesel = diesel_price / 9.8 


# --- C. Spezifische Investitionskosten (€/Einheit) ---
# Wichtig für PyPSA Komponenten (capital_cost pro p_nom)
# Beispiel Batterie: 470€ für 2.4 kWh
capex_per_kwh = 469.99 / 2.4 
annuity_per_kwh = annuity(capex_per_kwh, 10)


# --- D. Austauschkosten & Inflation (Advanced) ---
# Berechnung für Geräte, die kürzer leben als der Projektzeitraum (20 Jahre)
inflation_rate = 1.03 # 3%
years_future = 10     # Austausch nach 10 Jahren

# 1. Preis in der Zukunft berechnen
future_price = current_price * (inflation_rate ** years_future)

# 2. Annuität dieses zukünftigen Preises berechnen
future_annuity = annuity(future_price, lifetime_new_device)


# --- A. Lasten (Loads) ---
# 1. Statisch (konstant über das Jahr)
n.add("Load", "Starlink", bus="electricity", p_set=0.06)

# 2. Dynamisch (Zeitreihe aus CSV)
n.add("Load", "Haushalt", bus="electricity", p_set=df["Last_Profil"])


# --- B. Generatoren (Einspeiser) ---
# PV (Erneuerbar)
n.add("Generator", "PV", bus="electricity", 
      p_max_pu=df["PV_Erzeugung"], p_nom_extendable=True, capital_cost=...)

# Diesel (Brennstoff) -> In Network 2 als Supply
n.add("Generator", "Diesel_Tank", bus="diesel", 
      marginal_cost=mc_diesel, p_nom_extendable=True)


# --- C. Links (Umwandler) ---
# Wärmepumpe (Strom -> Wärme mit COP)
n.add("Link", "WP", bus0="electricity", bus1="thermal",
      efficiency=cop_time_series, p_max_pu=p_max_pu_series, p_nom_extendable=True)

# Diesel-Generator (Diesel -> Strom mit Wirkungsgrad)
n.add("Link", "GenSet", bus0="diesel", bus1="electricity",
      efficiency=0.35, p_nom_extendable=True)


# --- D. Speicher (StorageUnit) ---
# Batterie (Elektrisch)
n.add("StorageUnit", "Batterie", bus="electricity",
      max_hours=capacity/power,  # Verhältnis kWh zu kW
      efficiency_store=0.9,      # Ladeverlust
      efficiency_dispatch=0.95,  # Entladeverlust
      cyclic_state_of_charge=True) # Füllstand Ende = Füllstand Anfang

# Warmwasserspeicher (Thermisch mit Verlusten)
n.add("StorageUnit", "Boiler", bus="hot_water",
      standing_loss=0.02,        # 2% Verlust pro Stunde
      efficiency_store=1.0)


# --- A. Optimierung starten ---
n.optimize(solver_name="gurobi", threads=1, method=2)

# --- B. Dimensionierung abrufen (Was wurde gebaut?) ---
# p_nom_opt ist die optimierte Größe
pv_size_kw = n.generators.p_nom_opt["pv"]
bat_size_kwh = n.storage_units.p_nom_opt["batteriespeicher"] * n.storage_units.max_hours

# --- C. Zeitreihen abrufen (Was ist wann passiert?) ---
# _t steht für "time-dependent"
pv_generation_t = n.generators_t.p["pv"]       # Erzeugung PV (Zeitreihe)
battery_flow_t = n.storage_units_t.p["batterie"] # Laden/Entladen
load_t = n.loads_t.p["electrical_load"]          # Verbrauch

# --- D. Gesamtkosten Berechnung ---
# 1. Investitionskosten (Summe aller p_nom_opt * capital_cost)
capex_ges = (n.generators.p_nom_opt * n.generators.capital_cost).sum() + \
            (n.links.p_nom_opt * n.links.capital_cost).sum() + \
            (n.storage_units.p_nom_opt * n.storage_units.capital_cost).sum()

# 2. Betriebskosten (Summe aller Energieflüsse * Grenzkosten)
opex_ges = (n.generators_t.p.sum() * n.generators.marginal_cost).sum()

total_cost_per_year = capex_ges + opex_ges


# --- A. Einfacher Plot (Pandas integriert) ---
n.generators_t.p.plot() # Plottet alle Generatoren
plt.show()

# --- B. Komplexer Subplot (Wochendarstellung) ---
start = 168 # Stunde
end = 336   # Stunde

fig, ax = plt.subplots(figsize=(10, 5))

# Daten auswählen und plotten
n.loads_t.p["electrical_load"][start:end].plot(ax=ax, label="Last", color="red")
n.generators_t.p["pv"][start:end].plot(ax=ax, label="PV", color="yellow")

# Batterie: Negativ darstellen für Entladen?
# n.storage_units_t.p gibt pos. Werte beim Entladen, neg. beim Laden (Standard PyPSA Convention checken, variiert manchmal je nach Setup)

ax.set_ylabel("Leistung [kW]")
ax.set_title("Wochenprofil")
ax.legend()
ax.grid(True)
plt.show()

# --- C. Diesel Generator Invertieren (Trick aus Code 2) ---
# Damit er im Diagramm "unten" oder "oben" erscheint wie gewünscht
diesel_flow = n.links_t.p1["diesel_generator"][start:end] 
(-diesel_flow).plot(ax=ax, label="Diesel Gen")