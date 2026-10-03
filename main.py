import numpy as np
import matplotlib.pyplot as plt


# Parametrii de intrare algoritm genetic
M = 30          # dimensiunea populației
G_MAX = 60     # numărul maxim de generații
PC = 0.8        # probabilitatea de încrucișare
PM = 0.2       # probabilitatea de mutație
DS = 3          # dimensiunea turneului
W1 = 0.6        # pondere cost energetic
W2 = 0.2        # pondere confort
W3 = 0.2      # pondere PAR

# FLAGS - utilizate pentru greedy
RULEAZA_OPERATORI  = True   # tabelul combinații
RULEAZA_GRID       = True   # grid search hiperparametri
AFISEAZA_GRAFICE   = True   # graficele finale

######## Scenariu Exemplu
# # Informatii dispozitive (Exemplu)
# dispozitive = [
#     "Frigider",         # base
#     "Sistem alarma",    # base
#     "Masina de spalat", # neîntreruptibil
#     "Masina vase",      # neîntreruptibil
#     "Cuptor electric",  # neîntreruptibil
#     "Aer conditionat",  # întreruptibil
#     "Aspirator",        # întreruptibil
#     "TV",               # întreruptibil
#     "Laptop",           # întreruptibil
#     "Incarcator"        # întreruptibil
# ]
#
# N = len(dispozitive)   # numărul de dispozitive
#
# # Puterea nominală (kW)
# PN = np.array([0.15, 0.1, 2.1, 1.8, 2.5, 0.9, 0.6, 0.2, 0.1, 0.05])
# #PN = np.array([0.03, 0.04, 0.75, 0.85, 1.2, 0.5, 0.8, 0.08, 0.03, 0.015])
#
# # Tipul dispozitivelor: 0 = base, 1 = neîntreruptibil, 2 = întreruptibil
# TP = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
#
# # Numar utilizari zilnice
# NU = np.array([24, 24, 2, 3, 1, 4, 3, 6, 8, 4])
#
# # Durata minimă de funcționare (ore) — (pentru neîntreruptibile)
# DF = np.array([0, 0, 3, 2, 2, 0, 0, 0, 0, 0])
#
# # PREȚUL ENERGIEI ELECTRICE (lei/kWh) — pe ore (0-23)
# # Prețurile sunt mai mari în perioadele de activitate și mai mici în cele mai puțin active
# cost = np.array([          # ore de funcționare
#     0.4, 0.4, 0.4, 0.4,   # 0-3
#     0.4, 0.5, 0.6, 0.7,   # 4-7
#     0.6, 0.5, 0.3, 0.3,   # 8-11
#     0.3, 0.3, 0.4, 0.6,   # 12-15
#     0.9, 1.2, 1.3, 1.3,   # 16-19
#     1.2, 1.1, 0.8, 0.5    # 20-23
# ])
#
#
# # cost = np.array([
# #     1.37, 1.30, 1.27, 1.25, 1.27, 1.35,  # 0-5
# #     1.38, 1.29, 1.11, 0.83, 0.66, 0.67,  # 6-11
# #     0.62, 0.59, 0.60, 0.77, 1.01, 1.20,  # 12-17
# #     1.50, 1.58, 1.83, 1.73, 1.59, 1.61   # 18-23
# # ])
#
# ## vector preturi vara
# cost = np.array([
#     1.32, 1.28, 1.26, 1.25, 1.26, 1.29,  # 00 - 05
#     1.40, 1.43, 1.40, 1.28, 1.18, 1.11,  # 06 - 11
#     1.09, 0.98, 0.90, 0.94, 1.09, 1.25,  # 12 - 17
#     1.37, 1.49, 1.50, 1.44, 1.36, 1.34   # 18 - 23
#
# ])
# ##
# ## vector preturi iarna
# cost = np.array([
#     1.29, 1.28, 1.28, 1.25, 1.26, 1.35,  # 00 - 05
#     1.55, 1.65, 1.60, 1.52, 1.44, 1.36,  # 06 - 11
#     1.34, 1.36, 1.40, 1.59, 1.75, 1.85,  # 12 - 17
#     1.79, 1.71, 1.63, 1.55, 1.46, 1.35   # 18 - 23
# ])
# ##
#
# # MATRICEA DE CONFORT (N x 24)
# # Valoare mare = afectare mai mare a confortului utilizatorului
# # Valoare mică = afectare mai mică a confortului utilizatorului
#
# Conf = np.array([
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Frigider
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Sistem Camere
#     [2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  2],# Masina de spalat
#     [2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  2,  2,  2,  2],# Masina de vase
#     [2,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  0,  1,  1,  2,  2],# Cuptor electric
#     [1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],# Aer Conditionat
#     [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  1,  2,  2,  2],# Aspirator
#     [1,  1,  1,  2,  2,  2,  2,  1,  1,  1,  2,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1],# TV
#     [0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1],# Laptop
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0] # Incarcator
# ])

### Scenariu vara
# ##########
dispozitive = [
    "Frigider",                   # base
    "Sistem alarmă",              # base
    "Mașina de spălat haine",     # neîntreruptibil
    "Mașina de spălat vase",      # neîntreruptibil
    "Cuptor electric",            # neîntreruptibil
    "Air Fryer",                  # neîntreruptibil
    "Stație încărcare EV",        # întreruptibil
    "Sistem de irigații",         # întreruptibil
    "Pompă apă piscină",          # întreruptibil
    "Boiler electric",            # întreruptibil
    "Aer condiționat",            # întreruptibil
    "Aspirator",                  # întreruptibil
    "TV",                         # întreruptibil
    "Laptop",                     # întreruptibil
    "Încărcător telefon"          # întreruptibil
]

N = len(dispozitive)   # numărul de dispozitive

# Puterea nominală (kW)
PN = np.array([0.22, 0.20, 2.20, 1.35, 3.40, 1.50, 7.00, 0.75, 0.50, 6.00, 1.50, 0.70, 0.20, 0.15, 0.05])
#PN = np.array([0.03, 0.04, 0.75, 0.85, 1.2, 0.5, 0.8, 0.08, 0.03, 0.015])
#PN = np.array([0.15, 0.10, 2.10, 1.80, 2.10, 1.50, 5.50, 0.75, 0.50, 1.95, 1.10, 0.70, 0.20, 0.15, 0.05])
# Tipul dispozitivelor: 0 = base, 1 = neîntreruptibil, 2 = întreruptibil
TP = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# Numar utilizari zilnice
NU = np.array([24, 24, 2, 3, 1, 2, 4, 2, 4, 2, 6, 3, 4, 6, 4])

# Durata minimă de funcționare (ore) — (pentru neîntreruptibile)
DF = np.array([0, 0, 2, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ## vector preturi vara
cost = np.array([
    1.32, 1.28, 1.26, 1.25, 1.26, 1.29,  # 00 - 05
    1.40, 1.43, 1.40, 1.28, 1.18, 1.11,  # 06 - 11
    1.02, 0.94, 0.90, 0.94, 1.09, 1.25,  # 12 - 17
    1.37, 1.49, 1.50, 1.44, 1.36, 1.34   # 18 - 23
])
#cost = cost * 0.75
# MATRICEA DE CONFORT (N x 24)
# Valoare mare = afectare mai mare a confortului utilizatorului
# Valoare mică = afectare mai mică a confortului utilizatorului

Conf = np.array([
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Frigider
    [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Sistem alarme
    [2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2,  2],# Masina de spalat
    [2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  2],# Masina de vase
    [2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  2],# Cuptor electric
    [2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2],# Air Fryer
    [1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  1,  1,  1,  1],# Statie incarcare EV
    [1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0],# Sistem de irigatii
    [2,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  1,  1,  1,  2,  2],# Pompa filtrare piscina
    [2,  2,  2,  2,  1,  0,  0,  0,  1,  2,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0,  0,  1,  2,  2],# Boiler electric
    [1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],# Aer Conditionat
    [2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  2],# Aspirator
    [1,  1,  1,  2,  2,  2,  1,  0,  0,  1,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],# TV
    [0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],# Laptop
    [1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1] # Incarcator
])
#    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
##########

# ### Scenariu iaarna
# ##########
# ##########
# dispozitive = [
#     "Frigider",                # base
#     "Sistem alarmă",           # base
#     "Mașina de spălat haine",  # neîntreruptibil
#     "Mașina de spălat vase",   # neîntreruptibil
#     "Cuptor electric",         # neîntreruptibil
#     "Air Fryer",               # neîntreruptibil
#     "Stație încărcare EV",     # întreruptibil
#     "Sistem de încălzire",     # întreruptibil
#     "Radiator",                # întreruptibil
#     "Boiler electric",         # întreruptibil
#     "Aer condiționat",         # întreruptibil
#     "Aspirator",               # întreruptibil
#     "TV",                      # întreruptibil
#     "Laptop",                  # întreruptibil
#     "Încărcător telefon"       # întreruptibil
# ]
#
# N = len(dispozitive)   # numărul de dispozitive
#
# # Puterea nominală (kW)
# PN = np.array([0.22, 0.20, 2.20, 1.35, 3.40, 1.50, 7.00, 4.00, 0.75, 6.00, 1.50, 0.70, 0.20, 0.15, 0.05])
# # PN = np.array([0.15, 0.10, 2.10, 1.80, 2.10, 1.50, 5.50, 3.00, 1.75, 1.95, 1.10, 0.70, 0.20, 0.15, 0.05])
# #PN = np.array([0.03, 0.04, 0.75, 0.85, 1.2, 0.5, 0.8, 0.08, 0.03, 0.015])
#
# # Tipul dispozitivelor: 0 = base, 1 = neîntreruptibil, 2 = întreruptibil
# TP = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
#
# # Numar utilizari zilnice
# NU = np.array([24, 24, 1, 2, 1, 2, 3, 6, 2, 3, 1, 2, 6, 8, 8])
#
# # Durata minimă de funcționare (ore) — (pentru neîntreruptibile)
# DF = np.array([0, 0, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
# ## vector preturi iarna
# cost = np.array([
#     1.29, 1.28, 1.28, 1.25, 1.26, 1.35,  # 00 - 05
#     1.55, 1.65, 1.60, 1.52, 1.44, 1.36,  # 06 - 11
#     1.34, 1.36, 1.40, 1.59, 1.75, 1.85,  # 12 - 17
#     1.79, 1.71, 1.63, 1.55, 1.46, 1.35   # 18 - 23
# ])
# #
# # MATRICEA DE CONFORT (N x 24)
# # Valoare mare = afectare mai mare a confortului utilizatorului
# # Valoare mică = afectare mai mică a confortului utilizatorului
#
# Conf = np.array([
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Frigider
#     [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Sistem Camere
#     [2,  2,  2,  2,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  2,  2,  2],# Masina de spalat
#     [2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2,  2],# Masina de vase
#     [2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2],# Cuptor electric
#     [2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2],# Air Fryer
#     [1,  1,  1,  1,  1,  1,  2,  2,  2,  1,  0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  1,  1,  1,  1],# Statie incarcare EV
#     [1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  2,  2,  1,  1,  1,  2,  2,  2,  1,  1,  0,  0,  0,  0],# Sistem de incalzire
#     [1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  1],# Radiator
#     [1,  2,  2,  2,  1,  0,  0,  0,  0,  1,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0,  0,  1,  1,  1],# Boiler electric
#     [1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  1,  1,  1,  1],# Aer Conditionat
#     [2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  2,  2,  2,  2],# Aspirator
#     [1,  1,  1,  2,  2,  2,  1,  0,  0,  1,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1],# TV
#     [0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],# Laptop
#     [1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1] # Incarcator
# ])
# #    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
# # ##########
# ##########



# Afișare date de intrare
print("Configurație inițială")
print(f"Numărul maxim de generații: {G_MAX}")
print(f"Dimensiune populație/ Numărul de indivizi per generație: {M}")
print(f"Număr dispozitive: {N}")
print(f"Dispozitive: {dispozitive}")
print(f"Puterea nominală a dispozitivelor (kW): {PN}")
print(f"Tipuri de dispozitive (0=base, 1=neîntreruptibil, 2=întreruptibil): {TP}")
print(f"Durata minimă de funcționare (ore): {DF}")
print(f"Număr zilnic de utilizări a dispozitivelor: {NU}")
print(f"Ponderile W1 W2 W3: {W1}{W2:>5}{W3:>5}")
print("\nPrețul energiei electrice (lei/kWh)")
print(f"{'Ora'}", end="")
for h in range(24):
    print(f"{h:>5}", end="")
print()
print(f"{'Preț'}", end="")
for h in range(24):
    print(f"{cost[h]:>5}", end="")
print()
print("\nMatrice de confort (Dispozitiv)(Oră funcționare)")
print(f"{'Dispozitiv/Oră actv.':<22}", end="")
for h in range(24):
    print(f"{h:>3}", end="")
print()
for i in range(N):
    print(f"{dispozitive[i]:<22}", end="")
    for h in range(24):
        print(f"{Conf[i, h]:>3}", end="")
    print()

# ALgoritm genetic de optimizare - implemetare funcții
# Inițializare populație
def initializarea_populatiei(N, M, TP, NU, DF):
    populatie = []
    for k in range(M):
        Xk = np.zeros((N, 24), dtype=int)

        for i in range(N):
            if TP[i] == 0:          # continu/base — mereu activ pe 24 de ore
                Xk[i, :] = 1

            elif TP[i] == 1:        # neîntreruptibil
                ore_disponibile_n = list(range(24))
                for j in range(NU[i]): # de cate ori pe zi se folosesc in intervale de DF ore consecutive
                    pozitii_valide = []
                    # Verificare posibile puncte de start pentru executarea sarcinilor
                    for start in range(24 - DF[i] + 1):
                        interval_DF = list(range(start, start + DF[i]))
                        # Verificare disponibilitate ore interval selectat
                        if all(h in ore_disponibile_n for h in interval_DF):
                            pozitii_valide.append(start)
                    # Condiție de ieșire dacă nu există intervale disponibile
                    if len(pozitii_valide) == 0:
                        break
                    # Alegerea unei poziții de start care permit funcționarea unui dispozitiv pe un interval DF
                    ora_start = pozitii_valide[int(np.random.randint(0, len(pozitii_valide)))]
                    for m in range(ora_start, ora_start + DF[i]):
                       Xk[i, m] = 1
                       ore_disponibile_n.remove(m) # Marcare ore ocupate

            elif TP[i] == 2:        # întreruptibil
                ore_disponibile_i = list(range(24))
                nr_ore_active_i = min(NU[i],24) # Inițializare și asigurare funcționalitate
                ore_alese_i = np.random.choice(ore_disponibile_i, nr_ore_active_i, replace=False).tolist()
                for h in ore_alese_i:
                    Xk[i, h] = 1

        populatie.append(Xk)

    return populatie

# Determinare termeni fitness
def determinare_termeni_fitness(populatie, N, PN, Conf, cost):
    f_cost_v = []
    f_conf_v = []
    PAR_v = []
    for X in populatie:
        #Calculul consumului total - CT
        CT = np.zeros(24)
        for h in range(24):
            for i in range(N):
                CT[h] = CT[h] + PN[i] * X[i,h]

        # Calculul funcție de cost
        f_cost = 0
        for h in range(24):
            f_cost = f_cost + CT[h] * cost[h]

        # Calcului funcție de confort
        f_conf = 0
        for i in range(N):
            for h in range(24):
                f_conf = f_conf + X[i,h] * Conf[i,h]

        # Calcul PAR
        Pmax = np.max(CT)
        Pavg = (1/24) * np.sum(CT)
        PAR = Pmax / Pavg

        f_cost_v.append(f_cost)
        f_conf_v.append(f_conf)
        PAR_v.append(PAR)

    return f_cost_v, f_conf_v, PAR_v

# Evaluare fitness și Normalizare
def determinare_valori_maxime(N, PN, Conf, cost):
    # Se genereaza un individ cu totate valorie 1, caz maxim de consum, dar imposibil în practică
    x_max = np.ones((N, 24), dtype=int)
    CT_max = np.zeros(24)
    for h in range(24):
        for i in range(N):
            CT_max[h] = CT_max[h] + PN[i] * x_max[i, h]
    # Calculul f_cost maxim pentru consum CT maxim
    f_cost_max = 0
    for h in range(24):
        f_cost_max = f_cost_max + CT_max[h] * cost[h]
    # Calculul funcție de confort maximă
    f_conf_max = 0
    for i in range(N):
        for h in range(24):
            f_conf_max = f_conf_max + x_max[i, h] * Conf[i, h]
    # Determinare PAR_maxim
    x_PAR = np.zeros((N, 24), dtype=int)
    for h in range(24):
        for i in range(N):
            if h == 18:
                x_PAR[i, h] = 1
    CT_PAR = np.zeros(24)
    for h in range(24):
        for i in range(N):
            CT_PAR[h] = CT_PAR[h] + PN[i] * x_PAR[i, h]
    Pmax = np.max(CT_PAR)
    Pavg = (1 / 24) * np.sum(CT_PAR)
    PAR_max = Pmax / Pavg

    return f_cost_max, f_conf_max, PAR_max

def evaluare_fitness(f_cost, f_conf, PAR, W1, W2, W3, f_cost_maxim, f_confort_maxim, PAR_maxim):
    #Normalizare componente fitness
    f_cost_n = np.array(f_cost) / f_cost_maxim
    f_conf_n = np.array(f_conf) / f_confort_maxim
    PAR_n = np.array(PAR) / PAR_maxim
    #Calcul fitness
    fitness = W1 * f_cost_n + W2 * f_conf_n + W3 * PAR_n
    return fitness

#Selectie
# 1.Folosind metoda turneului
def selectie(P, fitness, M, DS):
    S = []
    for i in range(M):
        turneu = np.random.choice(len(P), DS, replace=False)
        castigator_index = turneu[0]
        castigator_fitness = fitness[turneu[0]]
        for j in range(DS):
            if fitness[turneu[j]] < castigator_fitness:
                castigator_index = turneu[j]
                castigator_fitness = fitness[turneu[j]]
        S.append(P[castigator_index].copy())
    return S

# 2. Selectia Rank
def selectie_rank(P, fitness, M, DS):
    S = []
    n = len(P)
    ordonare_indici = np.argsort(fitness)
    rang = np.zeros(n)
    for p, indx in enumerate(ordonare_indici):
        rang[indx] = n - p

    probabilitate = rang / np.sum(rang)

    for i in range(M):
        index = int(np.random.choice(n, p=probabilitate))
        S.append(P[index].copy())

    return S

# 3. Selectia Ruleata
def selectie_ruleta(P, fitness, M, DS):
    S = []
    n = len(P)
    fitness_arr = np.array(fitness)
    fitness_inv = np.max(fitness_arr) - fitness_arr

    if np.sum(fitness_inv) == 0:
        probabilitati = np.ones(n) / n
    else:
        probabilitati = fitness_inv / np.sum(fitness_inv)

    for i in range(M):
        index = int(np.random.choice(n, p=probabilitati))
        S.append(P[index].copy())

    return S

# Operația de Crossover
# 1. Crossover Single Point
def incrucisare(S, PC):
    P = []
    np.random.shuffle(S)
    for i in range(0, len(S), 2):
        p1 = S[i]
        p2 = S[i+1]
        numar_verficare = np.random.rand() #pentru comaprarea cu PC
        if numar_verficare < PC:
            punct_incrucisare = np.random.randint(1, 23)
            c1 = np.hstack([p1[:, :punct_incrucisare], p2[:, punct_incrucisare:]])
            c2 = np.hstack([p2[:, :punct_incrucisare], p1[:, punct_incrucisare:]])
        else:
            c1 = p1.copy()
            c2 = p2.copy()

        P.append(c1)
        P.append(c2)
    return P

# 2. Crossover Two-Point
def incrucisare_double(S, PC):
    P = []
    np.random.shuffle(S)
    for i in range(0, len(S), 2):
        p1 = S[i]
        p2 = S[i+1]
        numar_verficare = np.random.rand() #pentru comaprarea cu PC
        if numar_verficare < PC:
            punct_incrucisare = np.random.randint(1, 22)
            punct_incrucisare2 = np.random.randint(punct_incrucisare + 1, 23)
            c1 = np.hstack([p1[:, :punct_incrucisare], p2[:, punct_incrucisare: punct_incrucisare2], p1[:, punct_incrucisare2:]])
            c2 = np.hstack([p2[:, :punct_incrucisare], p1[:, punct_incrucisare: punct_incrucisare2], p2[:, punct_incrucisare2:]])
        else:
            c1 = p1.copy()
            c2 = p2.copy()

        P.append(c1)
        P.append(c2)
    return P

# 3. Crossover la nivel de bit
def incrucisare_bit(S, PC):
    P = []
    np.random.shuffle(S)
    for i in range(0, len(S), 2):
        p1 = S[i]
        p2 = S[i+1]
        p1_v = p1.flatten()
        p2_v = p2.flatten()
        numar_verficare = np.random.rand() #pentru comaprarea cu PC
        if numar_verficare < PC:
            punct_incrucisare = np.random.randint(1, N * 24 - 1)
            c1_v = np.concatenate([p1_v[:punct_incrucisare], p2_v[punct_incrucisare:]])
            c2_v = np.concatenate([p2_v[:punct_incrucisare], p1_v[punct_incrucisare:]])
        else:
            c1_v = p1_v.copy()
            c2_v = p2_v.copy()

        c1 = c1_v.reshape(N, 24)
        c2 = c2_v.reshape(N, 24)

        P.append(c1)
        P.append(c2)
    return P

# 4 . Crossover Uniform
def incrucisare_uniform(S, PC):
    P = []
    np.random.shuffle(S)
    for i in range(0, len(S), 2):
        p1 = S[i]
        p2 = S[i+1]
        c1 = p1.copy()
        c2 = p2.copy()
        if np.random.rand() < PC:
            for h in range(24):
                if np.random.rand() < 0.5:
                    c1[:, h] = p2[:, h]
                    c2[:, h] = p1[:, h]
        P.append(c1)
        P.append(c2)
    return P

# Operația de mutație
# 1. Mutaite Swap
def mutatie(P, PM, N, M, TP):
    P_mutatie = [i.copy() for i in P]
    for i in range(M):
        if np.random.rand() < PM:

            dispozitive_disponibile = [i for i in range(N) if TP[i] != 0]
            dispozitiv_ales = dispozitive_disponibile[int(np.random.randint(0, len(dispozitive_disponibile)))]

            h1 = np.random.randint(0, 24)
            h2 = np.random.randint(0, 24)

            val_h1 = P_mutatie[i][dispozitiv_ales, h1]
            P_mutatie[i][dispozitiv_ales, h1] = P_mutatie[i][dispozitiv_ales, h2]
            P_mutatie[i][dispozitiv_ales, h2] = val_h1

    return P_mutatie

# 2. Mutaite Bit Flip
def mutatie_bit(P, PM, N, M, TP):
    P_mutatie = [i.copy() for i in P]
    for i in range(M):
        for j in range(N):
            for h in range(24):
                numar_verficare = np.random.rand()
                if numar_verficare < PM:
                    P_mutatie[i][j, h] = 1 - P_mutatie[i][j, h]

    return P_mutatie


# 3. Mutatie Scramble
def mutatie_scramble(P, PM, N, M, TP):
    P_mutatie = [i.copy() for i in P]
    for i in range(M):
        if np.random.rand() < PM:
            dispozitive_disponibile = [j for j in range(N) if TP[j] != 0]
            dispozitiv_ales = dispozitive_disponibile[int(np.random.randint(0, len(dispozitive_disponibile)))]
            h1 = int(np.random.randint(0, 22))
            h2 = int(np.random.randint(h1+1, 24))
            interval = list(range(h1, h2))
            np.random.shuffle(interval)
            P_mutatie[i][dispozitiv_ales, h1:h2] = P_mutatie[i][dispozitiv_ales, interval]
    return P_mutatie

# 4. Mutatie Inversion/Reversing
def mutatie_inversion(P, PM, N, M, TP):
    P_mutatie = [individ.copy() for individ in P]
    for i in range(M):
        if np.random.rand() < PM:
            dispozitive_disponibile = [j for j in range(N) if TP[j] != 0]
            dispozitiv_ales = dispozitive_disponibile[int(np.random.randint(0, len(dispozitive_disponibile)))]
            h1 = int(np.random.randint(0, 22))
            h2 = int(np.random.randint(h1 + 1, 24))

            P_mutatie[i][dispozitiv_ales, h1:h2] = P_mutatie[i][dispozitiv_ales, h1:h2][::-1]
    return P_mutatie
# #

# Implementare constrângeri impuse algoritm
def implementare_constrangeri(P, N, TP, DF, NU, Conf, cost):
    P_cons = [i.copy() for i in P]
    for k in range(len(P_cons)):
        for i in range(N):
            if TP[i] == 0:
                P_cons[k][i, :] = 1
            elif TP[i] == 1:
                ore_active_1 = [h for h in range(24) if P_cons[k][i, h] == 1]
                P_cons[k][i, :] = 0
                nr_utilizare = 0
                verificare_interval = 0

                for start in ore_active_1:
                    if nr_utilizare < NU[i] and start >= verificare_interval:
                        if start + DF[i] <= 24:
                            P_cons[k][i, start : start + DF[i]] = 1
                            nr_utilizare += 1
                            verificare_interval = start + DF[i]

                if nr_utilizare < NU[i]:
                    ferestre_disponibile = [] # identifica ce ferestre au mai ramas libere dupa reconstructia int. de 1
                    for h in range(24 - DF[i] + 1):
                        if np.all(P_cons[k][i, h : h + DF[i]] == 0):
                            cost_fer = np.sum(cost[h : h + DF[i]])
                            conf_fer = np.sum(Conf[i, h : h + DF[i]])
                            scor_fer = conf_fer

                            ferestre_disponibile.append((scor_fer, h))

                    ferestre_disponibile.sort(key=lambda x: x[0])
                    for scor, h_start in ferestre_disponibile:
                        if nr_utilizare >= NU[i]:
                            break
                        if np.all(P_cons[k][i, h_start : h_start + DF[i]] == 0):
                            P_cons[k][i, h_start: h_start + DF[i]] = 1
                            nr_utilizare += 1

            elif TP[i] == 2:
                ore_active = int(np.sum(P_cons[k][i, :]))

                if ore_active > NU[i] * 1.3:
                    nr_ore_out = ore_active - NU[i]
                    ore_active_l = [h for h in range(24) if P_cons[k][i, h] == 1]
                    ore_active_l.sort(key=lambda h: Conf[i, h], reverse=True)
                    for h in ore_active_l[:nr_ore_out]:
                        P_cons[k][i, h] = 0
                elif ore_active < NU[i] * 0.7:
                    nr_ore_in = NU[i] - ore_active
                    ore_inactive_l = [h for h in range(24) if P_cons[k][i, h] == 0]
                    ore_inactive_l.sort(key=lambda h:  Conf[i, h])
                    for h in ore_inactive_l[:nr_ore_in]:
                        P_cons[k][i, h] = 1

    return P_cons

# Generare matrice baseline/ pentru comparatiia initial vs final
def genereaza_baseline(N, TP, NU, DF, Conf):
    baseline = np.zeros((N, 24), dtype=int)

    for i in range(N):

        if TP[i] == 0:
            baseline[i, :] = 1

        elif TP[i] == 1:
            ore_disponibile = list(range(24))

            for u in range(NU[i]):
                intervale_scor = []
                for start in range(24 - DF[i] + 1):
                    interval = list(range(start, start + DF[i]))
                    if all(h in ore_disponibile for h in interval):
                        scor = sum(Conf[i, h] for h in interval)
                        intervale_scor.append((scor, start))

                if len(intervale_scor) == 0:
                    break

                intervale_scor.sort(key=lambda x: x[0])
                ora_start = intervale_scor[0][1]

                for h in range(ora_start, ora_start + DF[i]):
                    baseline[i, h] = 1
                    ore_disponibile.remove(h)

        elif TP[i] == 2:
            scoruri_ore = [(Conf[i, h], h) for h in range(24)]
            scoruri_ore.sort(key=lambda x: x[0])
            ore_alese = [h for _, h in scoruri_ore[:NU[i]]]
            for h in ore_alese:
                baseline[i, h] = 1

    return baseline
# FUNCTII AFISARE GRAFICE SI PARAMETRII
def afisare_fitness_minim(arhiva_min, G_MAX):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(arhiva_min)), arhiva_min, color='blue', linewidth=2)
    # ← range(len(arhiva_min)) în loc de range(G_MAX + 1)
    plt.xlabel('Generație')
    plt.ylabel('Fitness minim')
    plt.title('Evoluția fitness-ului minim pe generații')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def afisare_fitness_mediu(arhiva_med, G_MAX):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(arhiva_med)), arhiva_med, color='blue', linewidth=2)
    plt.xlabel('Generație')
    plt.ylabel('Fitness mediu')
    plt.title('Evoluția fitness-ului mediu pe generații')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def afisare_constrangeri_confort(individ, Conf, N, dispozitive):
    constrangeri_per_ora = np.zeros(24, dtype=int)
    for h in range(24):
        for i in range(N):
            if individ[i, h] == 1 and Conf[i, h] == 2:
                constrangeri_per_ora[h] += 1

    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(24), constrangeri_per_ora,
                   color='steelblue', alpha=0.7, edgecolor='navy')
    val_max = int(np.max(constrangeri_per_ora))
    plt.ylim(0, val_max + 2)
    plt.yticks(range(0, val_max + 3))
    plt.xlabel('Ora')
    plt.ylabel('Număr dispozitive cu disconfort maxim')
    plt.title('Constrângeri de confort încălcate per oră\n(dispozitive active cu Conf=2)')
    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    for h, val in enumerate(constrangeri_per_ora):
        plt.text(h, val + 0.1, str(val), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

def afisare_planificare_gantt(individ, TP, dispozitive, N, titlu, fitness_val=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    culori = {0: 'steelblue', 1: 'tomato', 2: 'green'}
    etichete_tip = {0: 'Base', 1: 'Neîntreruptibil', 2: 'Întreruptibil'}
    tip_adaugat = set()

    for i in range(N):
        for h in range(24):
            if individ[i, h] == 1:
                culoare = culori[TP[i]]
                eticheta = etichete_tip[TP[i]] if TP[i] not in tip_adaugat else ""
                ax.barh(i, 1, left=h, color=culoare, edgecolor='white',
                        linewidth=0.5, label=eticheta, height=0.6)
                tip_adaugat.add(TP[i])

    ax.set_yticks(range(N))
    ax.set_yticklabels(dispozitive, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.set_xticks(range(25))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(25)], rotation=45, fontsize=8)
    ax.set_xlabel('Ora')

    if fitness_val is not None:
        ax.set_title(f'{titlu}\nFitness: {fitness_val:.4f}')
    else:
        ax.set_title(titlu)

    ax.grid(True, alpha=0.3)

    handles = [
        plt.Rectangle((0,0), 1, 1, color='steelblue', label='Continue'),
        plt.Rectangle((0,0), 1, 1, color='tomato', label='Neîntreruptibile'),
        plt.Rectangle((0,0), 1, 1, color='green', label='Întreruptibile')
    ]
    ax.legend(handles=handles, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()

#Afisare si calcul curbe de consum si de costuri
def calcul_CT(individ, PN, N):
    CT = np.zeros(24)
    for h in range(24):
        for i in range(N):
            CT[h] += PN[i] * individ[i, h]
    return CT

def afisare_consum_orar(CT1, CT2, label1, label2, titlu):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(24)
    width = 0.35
    ax.bar(x - width/2, CT1, width, label=label1, color='red', alpha=0.7)
    ax.bar(x + width/2, CT2, width, label=label2, color='blue', alpha=0.7)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, float(max(float(np.max(CT1)), float(np.max(CT2)))) + 2)
    ax.set_xlabel('Ora')
    ax.set_ylabel('Consum (kW)')
    ax.set_title(titlu)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def afisare_cost_orar(CT1, CT2, cost, label1, label2, titlu):
    cost_orar1 = CT1 * cost
    cost_orar2 = CT2 * cost
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(24)
    width = 0.35
    ax.bar(x - width/2, cost_orar1, width, label=label1, color='red', alpha=0.7)
    ax.bar(x + width/2, cost_orar2, width, label=label2, color='blue', alpha=0.7)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, float(max(float(np.max(cost_orar1)), float(np.max(cost_orar2)))) + 2)

    ax.set_xlabel('Ora')
    ax.set_ylabel('Cost orar (lei)')
    ax.set_title(titlu)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return CT1 * cost, CT2 * cost

def afisare_cost_total(cost_orar1, cost_orar2, label1, label2, titlu):
    cost1 = float(np.sum(cost_orar1))
    cost2 = float(np.sum(cost_orar2))
    economie = cost1 - cost2

    print(f"\nCost total {label1}: {cost1:.2f} lei")
    print(f"Cost total {label2}: {cost2:.2f} lei")
    print(f"Economie: {economie:.2f} lei")

    categorii = [f'Cost\n{label1}', f'Cost\n{label2}', 'Economie']
    valori = [cost1, cost2, economie]
    culori_bare = ['red', 'blue', 'green']

    plt.figure(figsize=(5, 4))
    bars = plt.bar(categorii, valori, color=culori_bare,
                   alpha=0.7, width=0.4, edgecolor='gray')
    for bar, val in zip(bars, valori):
        plt.text(bar.get_x() + bar.get_width()/2,
                 val + 0.3, f'{val:.2f} lei',
                 ha='center', va='bottom', fontsize=10)
    plt.ylim(0, float(max(cost1, cost2)) + 20)
    plt.ylabel('Cost total zilnic (lei)')
    plt.title(titlu)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(5, 4))
    # categorii = [f'Cost\n{label1}', f'Cost\n{label2}']
    # valori = [cost1, cost2]
    # culori_bare = ['red', 'blue']
    #
    # bars = plt.bar(categorii, valori, color=culori_bare,
    #                alpha=0.7, width=0.4, edgecolor='gray')
    # for bar, val in zip(bars, valori):
    #     plt.text(bar.get_x() + bar.get_width() / 2,
    #              val + 0.3, f'{val:.2f} lei',
    #              ha='center', va='bottom', fontsize=10)
    #
    # # Economie ca săgeată sau text
    # plt.annotate(f'Economie: {economie:.2f} lei',
    #              xy=(0.5, 0.95), xycoords='axes fraction',
    #              ha='center', fontsize=10,
    #              color='green', fontweight='bold')
    # plt.ylim(0, float(max(cost1, cost2)) + 20)
    # plt.ylabel('Cost total zilnic (lei)')
    # plt.title(titlu)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()


# Apelare funcții
P_pop = initializarea_populatiei(N, M, TP, NU, DF)
f_cost_v, f_conf_v, PAR_v = determinare_termeni_fitness(P_pop, N, PN, Conf, cost)
f_cost_maxim, f_confort_maxim, PAR_maxim = determinare_valori_maxime(N, PN, Conf, cost)
fitness = evaluare_fitness(f_cost_v, f_conf_v, PAR_v, W1, W2, W3, f_cost_maxim, f_confort_maxim, PAR_maxim)
S = selectie(P_pop, fitness, M, DS)
P_incrcucisare = incrucisare(S, PC)
P_mutatie = mutatie(P_incrcucisare, PM, N, M, TP)
#P_verificare= implementare_constrangeri(P_mutatie, N, TP, DF, NU, Conf, cost)

######### VERIFICARI
#VERIFICARE FUNCȚIONALITE FUNCTII - ETAPE  DE AFIȘARE
# Afisare populatie
print(f"\n Populație inițializată: {len(P_pop)} indivizi ")

for k in range(len(P_pop)):
    print(f"\nIndividul {k+1}")
    print(f"{'Dispozitiv/Oră actv.':<22}", end="")
    for h in range(24):
        print(f"{h:>3}", end="")
    print()
    for i in range(N):
        print(f"{dispozitive[i]:<22}", end="")
        for h in range(24):
            print(f"{P_pop[k][i,h]:>3}", end="")
        print()

# Afișare componente fitness
print("\nAfișare componente fitness - verificare calcule")
print(f"{'Individ':<10}{'f_cost':>10}{'f_conf':>10}{'PAR':>10}")
for k in range(len(P_pop)):
    print(f"Individu {k+1:<2}{f_cost_v[k]:>10.2f}{f_conf_v[k]:>10.2f}{PAR_v[k]:>10.2f}")

#Normalizare și Afișare fitness
print("\nCalcul/Evaluare/Afișare fitness")
# pentru verificare se vor afisa si valorile normalizate
print(f"{'Individ':<10}{'f_cost_nor.':>15}{'f_conf_nor.':>15}{'PAR_nor.':>15}{'fitness':>15}")
#fac o normalizare manuala pentru a pute verifica rezultatele
for k in range(len(P_pop)):
    f_cost_nor = f_cost_v[k] / f_cost_maxim
    f_conf_nor = f_conf_v[k] / f_confort_maxim
    PAR_nor = PAR_v[k] / PAR_maxim
    print(f"Individu {k+1:<2}{f_cost_nor:>12.2f}{f_conf_nor:>15.2f}{PAR_nor:>15.2f}{fitness[k]:>15.2f}")

# Selectie, Incrucisare, Mutatie
print("\nMecanismele și operatorii genetici")
#Verificare selecție
print("Verficare Selectie")
for i in range(len(S)):
    for j in range(len(P_pop)):
        if np.array_equal(S[i], P_pop[j]):
            print(f"Individ S {i+1:<6} → Individul P_pop {j+1:<3} fitness={fitness[j]:.2f}")
            break
print("\nAfisare populatie dupa selectie")
for k in range(len(S)):
    print(f"\nIndividul {k+1} Selectie")
    print(f"{'Dispozitiv/Oră actv.':<22}", end="")
    for h in range(24):
        print(f"{h:>3}", end="")
    print()
    for i in range(N):
        print(f"{dispozitive[i]:<22}", end="")
        for h in range(24):
            print(f"{S[k][i,h]:>3}", end="")
        print()

#Verificare încrucișare
print(f"\n\nRezultat Incrucisare")
for k in range(len(P_incrcucisare)):
    print(f"\nIndividul {k+1} incrucisare")
    print(f"{'Dispozitiv/Oră actv.':<22}", end="")
    for h in range(24):
        print(f"{h:>3}", end="")
    print()
    for i in range(N):
        print(f"{dispozitive[i]:<22}", end="")
        for h in range(24):
            print(f"{P_incrcucisare[k][i,h]:>3}", end="")
        print()

#Verificarea operației de mutație
print(f"\nEtapa de mutatie")
for k in range(len(P_mutatie)):
    print(f"\nIndividul {k+1} mutatie")
    print(f"{'Dispozitiv/Oră actv.':<22}", end="")
    for h in range(24):
        print(f"{h:>3}", end="")
    print()
    for i in range(N):
        print(f"{dispozitive[i]:<22}", end="")
        for h in range(24):
            print(f"{P_mutatie[k][i,h]:>3}", end="")
        print()

# Afișare date de ieșire după prima generație
individ_minim = P_pop[np.argmin(fitness)].copy()
fitness_minim = np.min(fitness)
generatie_minim = 0
index_minim = int(np.argmin(fitness))
print(f"\nFitness minimi: {fitness_minim:.2f}")
print(f"Individul { index_minim + 1}")
print(f"{'Dispozitiv/Oră actv.':<22}", end="")
for h in range(24):
    print(f"{h:>3}", end="")
print()
for i in range(N):
    print(f"{dispozitive[i]:<22}", end="")
    for h in range(24):
        print(f"{individ_minim[i,h]:>3}", end="")
    print()

arhiva_fitness_minim = [fitness_minim]
arhiva_fitness_mediu = [float (np.mean(fitness))]
P_pop_initial = [x.copy() for x in P_pop]
print(f"\nVerificare Adrese")
print(id(P_pop[0]))
print(id(S[0]))
print(id(P_incrcucisare[0]))
print(id(P_mutatie[0]))

#Bucla evolutivă din algoritmul genetic
# print(f"\nBucla evolutiva")
# for g in range(G_MAX):
#     #Elitism -partea 1
#     elitism_index = int(np.argmin(fitness))
#     elitism_individ = P_pop[elitism_index].copy()
#     elitism_fitness = float(fitness[elitism_index])
#
#     #Algoritm genetic
#     S_g = selectie(P_pop, fitness, M, DS)
#     Inc_g = incrucisare(S_g, PC)
#     P_mutatie_g = mutatie(Inc_g, PM, N, M, TP)
#     P_final = implementare_constrangeri(P_mutatie_g, N, TP, DF, NU, Conf, cost)
#     # Evaluare fitness
#     f_cost_v_g, f_conf_v_g, PAR_v_g = determinare_termeni_fitness(P_final, N, PN, Conf, cost)
#     fitness_g = evaluare_fitness(f_cost_v_g, f_conf_v_g, PAR_v_g, W1, W2, W3, f_cost_maxim, f_confort_maxim, PAR_maxim)
#     #elitism - partea 2
#     max_index = int(np.argmax(fitness_g))
#     P_final[max_index] = elitism_individ
#     fitness_g[max_index] = elitism_fitness
#
#     P_pop = [x.copy() for x in P_final] # Actualizare populatie pentru urmatoarea rulare
#     fitness = fitness_g
#     fitness_minim_g = float(np.min(fitness_g))
#     index_minim_g = int(np.argmin(fitness_g))
#     if fitness_minim_g < float(fitness_minim):
#         fitness_minim = fitness_minim_g
#         individ_minim = P_final[index_minim_g].copy()
#         generatie_minim = g+1
#         index_minim = index_minim_g
#     arhiva_fitness_minim.append(fitness_minim_g)
#     arhiva_fitness_mediu.append(float(np.mean(fitness_g)))
#     #Contorizare date/generatie
#     print(f"\n Generatia {g+1} Fitness minimi: {fitness_minim_g:.3f}")
#
# ## REZULTATE ALGORITM
# ## GENERARE BASELINE
# baseline = genereaza_baseline(N, TP, NU, DF, Conf)
#
# # Calculul componentelor fitness pentru baseline
# f_cost_b, f_conf_b, PAR_b = determinare_termeni_fitness(
#     [baseline], N, PN, Conf, cost)
# fitness_baseline = evaluare_fitness(
#     f_cost_b, f_conf_b, PAR_b,
#     W1, W2, W3, f_cost_maxim, f_confort_maxim, PAR_maxim)
#
# print(f"\n Profil baseline")
# print(f"f_cost:  {f_cost_b[0]:.2f} lei")
# print(f"f_conf:  {f_conf_b[0]:.2f}")
# print(f"PAR:     {PAR_b[0]:.4f}")
# print(f"Fitness: {fitness_baseline[0]:.4f}")
# print(f"\n{'Dispozitiv/Oră':<22}", end="")
# for h in range(24):
#     print(f"{h:>3}", end="")
# print()
# for i in range(N):
#     print(f"{dispozitive[i]:<22}", end="")
#     for h in range(24):
#         print(f"{baseline[i,h]:>3}", end="")
#     print()
#
# print(f"\nRezultat algoritm")
# print(f"Fitness minimi algotitm: {fitness_minim:.4f}")
# # O sa continuii dupa consulatii cand o sa am o idee despre toate datele de iesire stabilite
#
# # print(f"\nVerificare Adrese")
# # print(id(P_pop[0]))
# # print(id(S_g[0]))
# # print(id(Inc_g[0]))
# # print(id(P_final[0]))
# # print(id(fitness[0]))
# # print(id(fitness_g[0]))
#
# print(f"\nReturnare minime")
# print(f"Fitness minim global:  {fitness_minim:.5f}")
# print(f"Găsit în generația:    {generatie_minim}")
# print(f"Individul nr:          {index_minim + 1}")
# print(f"{'Dispozitiv/Oră':<22}", end="")
# for h in range(24):
#     print(f"{h:>3}", end="")
# print()
# for i in range(N):
#     print(f"{dispozitive[i]:<22}", end="")
#     for h in range(24):
#         print(f"{individ_minim[i,h]:>3}", end="")
#     print()
#
# #Generare matrice calcul PAR
# ##################AFISARE VALORI MAXIME
# # print(f"f_cost_maxim: {f_cost_maxim:.2f}")
# # print(f"f_conf_maxim: {f_confort_maxim:.2f}")
# # print(f"PAR_maxim: {PAR_maxim:.2f}")
#
# # P_verificare = initializarea_populatiei(N, M, TP, DF)
# # f_cost_vv, f_conf_vv, PAR_vv = determinare_componenete_fitness(P_verificare, N, PN, Conf, cost)
# # fitnessv = evaluare_fitness(f_cost_vv, f_conf_vv, PAR_vv, W1, W2, W3, f_cost_maxim, f_confort_maxim, PAR_maxim)
# # #Sv = selectie(P_verificare, fitnessv, M, DS)
# # P_incrcucisarev = incrucisare(P_verificare, PC, N)
# # P_mutatiev = mutatie(P_incrcucisarev, PM, N, M, TP)
# # print(f"\n Populație verificata: {len(P_verificare)} indivizi ")
# #
# # for k in range(len(P_verificare)):
# #     print(f"\nIndividul {k+1}")
# #     print(f"{'Dispozitiv/Oră actv.':<22}", end="")
# #     for h in range(24):
# #         print(f"{h:>3}", end="")
# #     print()
# #     for i in range(N):
# #         print(f"{dispozitive[i]:<22}", end="")
# #         for h in range(24):
# #             print(f"{P_verificare[k][i,h]:>3}", end="")
# #         print()
# #
# # print(f"\n Populație verificata dupa mutatie: {len(P_verificare)} indivizi ")
# #
# # for k in range(len(P_mutatiev)):
# #     print(f"\nIndividul {k+1}")
# #     print(f"{'Dispozitiv/Oră actv.':<22}", end="")
# #     for h in range(24):
# #         print(f"{h:>3}", end="")
# #     print()
# #     for i in range(N):
# #         print(f"{dispozitive[i]:<22}", end="")
# #         for h in range(24):
# #             print(f"{P_mutatiev[k][i,h]:>3}", end="")
# #         print()
# # pvc= implementare_constrangeri(P_mutatiev, N, TP, DF, NU, Conf, cost)
# #
# # print(f"\n Populație verificata dupa implemetare constrangeri: {len(P_verificare)} indivizi ")
# #
# # for k in range(len(pvc)):
# #     print(f"\nIndividul {k+1}")
# #     print(f"{'Dispozitiv/Oră actv.':<22}", end="")
# #     for h in range(24):
# #         print(f"{h:>3}", end="")
# #     print()
# #     for i in range(N):
# #         print(f"{dispozitive[i]:<22}", end="")
# #         for h in range(24):
# #             print(f"{pvc[k][i,h]:>3}", end="")
# #         print()
#
#
#
# ################################################################
# #
# #Selectare mecanism selectie si operatori genetici
#
# top5_operatori = []
# rezultate_finale = []
# arh_min = []
# arh_med = []
# fit_min_top = 1.0
#
#
# if RULEAZA_OPERATORI:
#     operatori = {
#         'selectie': {
#             'Turneu': selectie,
#             'Rank':   selectie_rank,
#             'Ruleta': selectie_ruleta
#         },
#         'crossover': {
#             'Single':  incrucisare,
#             'Double':  incrucisare_double,
#             'Bit':     incrucisare_bit,
#             'Uniform': incrucisare_uniform
#         },
#         'mutatie': {
#             'Swap':      (mutatie,           0.2),
#             'Bitflip':   (mutatie_bit,        0.02),
#             'Scramble':  (mutatie_scramble,   0.2),
#             'Inversion': (mutatie_inversion,  0.2)
#         }
#     }
#
#     NR_RULARI = 10
#     rezultate_op = []
#     total_op = 3 * 4 * 4
#     contor_op = 0
#
#     P_comun_op = initializarea_populatiei(N, M, TP, NU, DF)
#     f_c0, f_co0, par0 = determinare_termeni_fitness(P_comun_op, N, PN, Conf, cost)
#     fit_comun_op = evaluare_fitness(f_c0, f_co0, par0, W1, W2, W3,f_cost_maxim, f_confort_maxim, PAR_maxim)
#
#     for num_s, func_s in operatori['selectie'].items():
#         for num_c, func_c in operatori['crossover'].items():
#             for num_m, (func_m, pm_val) in operatori['mutatie'].items():
#                 contor_op += 1
#                 print(f"{contor_op}/{total_op}: {num_s}+{num_c}+{num_m}", end=" → ")
#
#                 fitness_rulari_op = []
#
#                 for rulare in range(NR_RULARI):
#                     P = [x.copy() for x in P_comun_op]
#                     fit = fit_comun_op.copy()
#                     fit_min = float(np.min(fit))
#
#                     for g in range(G_MAX):
#                         idx_elite = int(np.argmin(fit))
#                         elite = P[idx_elite].copy()
#                         fit_elite = float(fit[idx_elite])
#
#                         S = func_s(P, fit, M, DS)
#                         Pnou = func_c(S, PC)
#                         Pnou = func_m(Pnou, pm_val, N, M, TP)
#                         Pnou = implementare_constrangeri(Pnou, N, TP, DF, NU, Conf, cost)
#
#                         f_c, f_co, par = determinare_termeni_fitness(Pnou, N, PN, Conf, cost)
#                         fit = evaluare_fitness(f_c, f_co, par, W1, W2, W3,f_cost_maxim, f_confort_maxim, PAR_maxim)
#
#                         idx_slab = int(np.argmax(fit))
#                         Pnou[idx_slab] = elite
#                         fit[idx_slab] = fit_elite
#
#                         P = [x.copy() for x in Pnou]
#
#                         if float(np.min(fit)) < fit_min:
#                             fit_min = float(np.min(fit))
#
#                     fitness_rulari_op.append(fit_min)
#
#                 media_op = float(np.mean(fitness_rulari_op))
#                 print(f"media={media_op:.4f}")
#
#                 rezultate_op.append({
#                     'Selectie':  num_s,
#                     'Crossover': num_c,
#                     'Mutatie':   num_m,
#                     'func_s':    func_s,
#                     'func_c':    func_c,
#                     'func_m':    func_m,
#                     'pm_val':    pm_val,
#                     'Media':     round(media_op, 4),
#                     'Min':       round(min(fitness_rulari_op), 4),
#                     'Max':       round(max(fitness_rulari_op), 4),
#                     'Rulari':    [round(v, 4) for v in fitness_rulari_op]
#                 })
#
#     # Sortare și afișare
#     rezultate_op.sort(key=lambda x: x['Media'])
#
#     print(f"\n\n{'Nr':<4}{'Selecție':<12}{'Crossover':<12}{'Mutație':<12}"
#           f"{'Media':>10}{'Min':>10}{'Max':>10} Toate rulările")
#     print("-" * 90)
#     for idx, r in enumerate(rezultate_op):
#         rulari_str = "  ".join([f"{v:.4f}" for v in r['Rulari']])
#         print(f"{idx+1:<4}{r['Selectie']:<12}{r['Crossover']:<12}"
#               f"{r['Mutatie']:<12}{r['Media']:>10.4f}"
#               f"{r['Min']:>10.4f}{r['Max']:>10.4f}  [{rulari_str}]")
#     # Salvare Top 5
#     top5_operatori = rezultate_op[:5]
#     print(f"\n=== TOP 5 OPERATORI ===")
#     for r in top5_operatori:
#         print(f"{r['Selectie']}+{r['Crossover']}+{r['Mutatie']} "
#               f"→ media={r['Media']:.4f}")
#
# ################################################################
# ##############################################################
# # HIPERPARAMETRI
#
# NR_RULARI_HP = 10
# populatii_comun = {}
# for hp_M_init in [30, 40, 50]:
#     P_init = initializarea_populatiei(N, hp_M_init, TP, NU, DF)
#     f_c_i, f_co_i, par_i = determinare_termeni_fitness(P_init, N, PN, Conf, cost)
#     fit_init = evaluare_fitness(f_c_i, f_co_i, par_i, W1, W2, W3,f_cost_maxim, f_confort_maxim, PAR_maxim)
#     populatii_comun[hp_M_init] = (P_init, fit_init)
#
# if RULEAZA_GRID:
#     import itertools
#
#     grid_M =  [30, 40, 50]
#     grid_G =  [40, 60, 80]
#     grid_PC = [0.7, 0.75, 0.8, 0.9]
#     grid_PM = [0.2, 0.25, 0.3, 0.35]
#
#     combinatii_hp = list(itertools.product(grid_M, grid_G, grid_PC, grid_PM))
#     rezultate_finale = []
#     total_grid = len(top5_operatori) * len(combinatii_hp)
#     contor_grid = 0
#
#     for op in top5_operatori:
#         for (hp_M, hp_G, hp_PC, hp_PM) in combinatii_hp:
#             contor_grid += 1
#             print(f"{contor_grid}/{total_grid}: "
#                   f"{op['Selectie']}+{op['Crossover']}+{op['Mutatie']} "
#                   f"M={hp_M} G={hp_G} PC={hp_PC} PM={hp_PM}",
#                   end=" → ")
#
#             fitness_rulari_hp = []
#
#             for rulare in range(NR_RULARI_HP):
#                 arh_min_r, arh_med_r = [], []
#
#                 P_baza, fit_baza = populatii_comun[hp_M]
#                 P = [x.copy() for x in P_baza]
#                 fit = fit_baza.copy()
#
#                 arh_min_r.append(float(np.min(fit)))
#                 arh_med_r.append(float(np.mean(fit)))
#                 fit_min_r = float(np.min(fit))
#                 individ_r = P[np.argmin(fit)].copy()
#
#                 for g in range(hp_G):
#                     idx_elite = int(np.argmin(fit))
#                     elite = P[idx_elite].copy()
#                     fit_elite = float(fit[idx_elite])
#
#                     S = op['func_s'](P, fit, hp_M, DS)
#                     Pnou = op['func_c'](S, hp_PC)
#                     Pnou = op['func_m'](Pnou, hp_PM, N, hp_M, TP)
#                     Pnou = implementare_constrangeri(Pnou, N, TP, DF, NU, Conf, cost)
#
#                     f_c, f_co, par = determinare_termeni_fitness(Pnou, N, PN, Conf, cost)
#                     fit = evaluare_fitness(f_c, f_co, par, W1, W2, W3,f_cost_maxim, f_confort_maxim, PAR_maxim)
#
#                     idx_slab = int(np.argmax(fit))
#                     Pnou[idx_slab] = elite
#                     fit[idx_slab] = fit_elite
#
#                     P = [x.copy() for x in Pnou]
#
#                     arh_min_r.append(float(np.min(fit)))
#                     arh_med_r.append(float(np.mean(fit)))
#
#                     if float(np.min(fit)) < fit_min_r:
#                         fit_min_r = float(np.min(fit))
#                         individ_r = P[np.argmin(fit)].copy()
#
#                 fitness_rulari_hp.append(fit_min_r)
#
#
#                 if len(fitness_rulari_hp) == 1 or fit_min_r <= min(fitness_rulari_hp[:-1]):
#                     arh_min_salvata = arh_min_r.copy()
#                     arh_med_salvata = arh_med_r.copy()
#                     individ_salvat = individ_r.copy()
#                     fit_min_salvat = fit_min_r
#
#             media_hp = float(np.mean(fitness_rulari_hp))
#             print(f"media={media_hp:.4f}")
#
#             rezultate_finale.append({
#                 'Selectie': op['Selectie'],
#                 'Crossover': op['Crossover'],
#                 'Mutatie': op['Mutatie'],
#                 'func_s': op['func_s'],
#                 'func_c': op['func_c'],
#                 'func_m': op['func_m'],
#                 'M': hp_M, 'G': hp_G,
#                 'PC': hp_PC, 'PM': hp_PM,
#                 'Media': round(media_hp, 4),
#                 'Min': round(min(fitness_rulari_hp), 4),
#                 'Max': round(max(fitness_rulari_hp), 4),
#                 'Rulari': [round(v, 4) for v in fitness_rulari_hp],
#                 'arh_min': arh_min_salvata,
#                 'arh_med': arh_med_salvata,
#                 'individ': individ_salvat,
#                 'fit_min': fit_min_salvat
#             })
#
#     rezultate_finale.sort(key=lambda x: x['Media'])
#
#     print(f"\n{'Nr':<4}{'Sel':<10}{'Cross':<10}{'Mut':<12}"
#           f"{'M':>5}{'G':>5}{'PC':>6}{'PM':>6}"
#           f"{'Media':>10}{'Min':>10}")
#     print("-" * 78)
#     for idx, r in enumerate(rezultate_finale[:30]):
#         rulari_str = "  ".join([f"{v:.4f}" for v in r['Rulari']])
#         print(f"{idx + 1:<4}{r['Selectie']:<10}{r['Crossover']:<10}"
#               f"{r['Mutatie']:<12}{r['M']:>5}{r['G']:>5}"
#               f"{r['PC']:>6}{r['PM']:>6}"
#               f"{r['Media']:>10.4f}{r['Min']:>10.4f}  [{rulari_str}]")
#
#     print(f"\nTOP 5 CONFIGURAȚII FINALE")
#     print(f"\n{'Nr':<4}{'Sel':<10}{'Cross':<10}{'Mut':<12}"
#           f"{'M':>5}{'G':>5}{'PC':>6}{'PM':>6}"
#           f"{'Media':>10}{'Min':>10}")
#     print("-" * 78)
#     for idx, r in enumerate(rezultate_finale[:5]):
#         print(f"{idx + 1:<4}{r['Selectie']:<10}{r['Crossover']:<10}"
#               f"{r['Mutatie']:<12}{r['M']:>5}{r['G']:>5}"
#               f"{r['PC']:>6}{r['PM']:>6}"
#               f"{r['Media']:>10.4f}{r['Min']:>10.4f}")
#
#     print(f"\nCele mai neperformante 5 CONFIGURAȚII")
#     print(f"\n{'Nr':<4}{'Sel':<10}{'Cross':<10}{'Mut':<12}"
#           f"{'M':>5}{'G':>5}{'PC':>6}{'PM':>6}"
#           f"{'Media':>10}{'Min':>10}")
#     print("-" * 78)
#     for idx, r in enumerate(rezultate_finale[-5:]):
#         print(f"{len(rezultate_finale) - 4 + idx:<4}{r['Selectie']:<10}{r['Crossover']:<10}"
#               f"{r['Mutatie']:<12}{r['M']:>5}{r['G']:>5}"
#               f"{r['PC']:>6}{r['PM']:>6}"
#               f"{r['Media']:>10.4f}{r['Min']:>10.4f}")
#
# #############################################################
# if AFISEAZA_GRAFICE:
#     for idx in range(5):
#         r = rezultate_finale[idx]
#
#         print(f"\nConfigurația {idx+1}")
#         print(f"{r['Selectie']}+{r['Crossover']}+{r['Mutatie']} "
#               f"M={r['M']} G={r['G']} PC={r['PC']} PM={r['PM']}")
#         print(f"Fitness minim: {r['fit_min']:.4f}")
#
#         afisare_fitness_minim(r['arh_min'], r['G'])
#         afisare_fitness_mediu(r['arh_med'], r['G'])
#         afisare_planificare_gantt(r['individ'], TP, dispozitive, N,
#                                   f"Varianta {idx+1}: Programarea optimă",
#                                   r['fit_min'])
#         afisare_constrangeri_confort(r['individ'], Conf, N, dispozitive)
#
#         CT_b = calcul_CT(baseline, PN, N)
#         CT_o = calcul_CT(r['individ'], PN, N)
#         afisare_consum_orar(CT_b, CT_o, 'Baseline', 'Optim',
#                             f'Varianta {idx+1}: Comparație consum')
#         cost_b, cost_o = afisare_cost_orar(CT_b, CT_o, cost,'Baseline', 'Optim',
#                                             f'Top {idx+1}: Cost orar')
#         afisare_cost_total(cost_b, cost_o, 'Baseline', 'Optim',
#                            f'Varianta {idx+1}: Cost total')
# #############################################################

# # TEST grafic fitnes mediu si minim
# plt.figure(figsize=(10, 5))
# plt.plot(range(G_MAX + 1), arhiva_fitness_minim, color='blue', linewidth=2)
# plt.xlabel('Generație')
# plt.ylabel('Fitness minim')
# plt.title('Evoluția fitness-ului minim pe generații')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
# #test grafic
# plt.figure(figsize=(10, 5))
# plt.plot(range(G_MAX + 1), arhiva_fitness_mediu, color='blue', linewidth=2)
# plt.xlabel('Generație')
# plt.ylabel('Fitness mediu')
# plt.title('Evoluția fitness-ului mediu pe generații')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

#GRAFICUL DE PETURI
# plt.figure(figsize=(15, 5))
# plt.plot(range(24), cost, color='red', linewidth=2, marker='o', markersize=5)
# plt.xlabel('Ora')
# plt.ylabel('Preț (lei/kWh)')
# plt.title('Prețul energiei electrice pe ore')
# plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# ## 3.GRAFIC CONSTRANGERI DE CONFORT INCALCATE PE ORA
# constrangeri_per_ora = np.zeros(24, dtype=int)
# for h in range(24):
#     for i in range(N):
#         if individ_minim[i, h] == 1 and Conf[i, h] == 2:
#             constrangeri_per_ora[h] += 1
# plt.figure(figsize=(12, 5))
# bars = plt.bar(range(24), constrangeri_per_ora, color='steelblue', alpha=0.7, edgecolor='navy')
# # Definire axa verticala
# val_max = int(np.max(constrangeri_per_ora))
# plt.ylim(0, val_max + 2)
# plt.yticks(range(0, val_max + 3))
# # Afisare
# plt.xlabel('Ora')
# plt.ylabel('Număr dispozitive cu disconfort maxim')
# plt.title('Constrângeri de confort încălcate per oră\n(dispozitive active cu Conf=2)')
# plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
# plt.grid(True, alpha=0.3, axis='y')
# # Valorile nule
# for h, val in enumerate(constrangeri_per_ora):
#     plt.text(h, val + 0.1, str(val), ha='center', va='bottom', fontsize=9)
# plt.tight_layout()
# plt.show()
#
# ## 4.PLANIFICAREA FUNCTIONALITATII INDIVIDULUI MINIM - DIAGRAMA GANTT
# ###
# fig, ax = plt.subplots(figsize=(14, 6))
#
# # Stabilire format per tip de functionare
# culori = {0: 'steelblue', 1: 'tomato', 2: 'green'}
# etichete_tip = {0: 'Base', 1: 'Neîntreruptibil', 2: 'Întreruptibil'}
# tip_adaugat = set()
#
# for i in range(N):
#     for h in range(24):
#         if individ_minim[i, h] == 1:
#             culoare = culori[TP[i]]
#             eticheta = etichete_tip[TP[i]] if TP[i] not in tip_adaugat else ""
#             ax.barh(i, 1, left=h, color=culoare, edgecolor='white', linewidth=0.5, label=eticheta, height=0.6)
#             tip_adaugat.add(TP[i])
#
# ax.set_yticks(range(N))
# ax.set_yticklabels(dispozitive, fontsize=10)
# ax.invert_yaxis()
# ax.set_xlim(0, 24)
# ax.set_xticks(range(25))
# ax.set_xticklabels([f'{h:02d}:00' for h in range(25)], rotation=45, fontsize=8)
#
# ax.set_xlabel('Ora')
# ax.set_title(f'Programarea optimă a dispozitivelor\nFitness minim: {fitness_minim:.4f}')
# ax.grid(True, alpha=0.3)
#
# # Legendă pentru tipuri
# handles = [
#     plt.Rectangle((0,0), 1, 1, color='steelblue', label='Continue'),
#     plt.Rectangle((0,0), 1, 1, color='tomato', label='Neîntreruptibile'),
#     plt.Rectangle((0,0), 1, 1, color='green', label='Întreruptibile')
# ]
# ax.legend(handles=handles, loc='upper left',bbox_to_anchor=(1.01, 1), borderaxespad=0)
# plt.tight_layout()
# plt.show()
# ###
# #5. Comparație costuri înainte/după
# # Calculul CT pentru populația inițială — media populației
# CT_initial = np.zeros(24)
# for X in P_pop_initial:
#     for h in range(24):
#         for i in range(N):
#             CT_initial[h] += PN[i] * X[i, h]
# CT_initial /= M
# # CT pentru individul optim
# CT_best = np.zeros(24)
# for h in range(24):
#     for i in range(N):
#         CT_best[h] += PN[i] * individ_minim[i, h]
#
# fig, ax = plt.subplots(figsize=(14, 5))
# x = np.arange(24)
# width = 0.35
# ax.bar(x - width/2, CT_initial, width, label='Înainte (medie populație inițială)', color='red', alpha=0.7)
# ax.bar(x + width/2, CT_best, width, label='După (individ optim)', color='blue', alpha=0.7)
# ax.set_xlim(-0.5, 23.5)
# max_consum = max(np.max(CT_initial), np.max(CT_best))
# ax.set_ylim(0, max_consum + 2)
# ax.set_xlabel('Ora')
# ax.set_ylabel('Consum (kW)')
# ax.set_title('Comparație profil de consum — înainte vs după optimizare')
# ax.set_xticks(x)
# ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
# ##
# #6. Curba de consum/energie — profilul CT(h) × cost(h)
# # Cost orar total = consum × preț
# cost_orar_initial = CT_initial * cost
# cost_orar_best = CT_best * cost
#
# fig1, ax1 = plt.subplots(figsize=(14, 5))
# x = np.arange(24)
# width = 0.35
# ax1.bar(x - width/2, cost_orar_initial, width, label='Cost caz initial', color='red', alpha=0.7)
# ax1.bar(x + width/2, cost_orar_best, width, label='Cost caz optim', color='blue', alpha=0.7)
# ax1.set_xlim(-0.5, 23.5)
# max_cost = max(np.max(cost_orar_initial), np.max(cost_orar_best))
# ax1.set_ylim(0, max_cost + 2 )
# ax1.set_xlabel('Ora')
# ax1.set_ylabel('Cost orar (lei)')
# ax1.set_title('Cost energetic orar — înainte vs după optimizare')
# ax1.set_xticks(x)
# ax1.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
# ax1.legend()
# ax1.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# # 7. GRAFIC EXPRIMARE COST TOTAL ININTE SI DUPA
# cost_total_inainte= np.sum(cost_orar_initial)
# cost_total_dupa = np.sum(cost_orar_best)
# print(f"\n\nCost total înainte: {cost_total_inainte:.2f} lei")
# print(f"Cost total după:    {cost_total_dupa:.2f} lei")
# economie = float(cost_total_inainte) - float(cost_total_dupa)
# print(f"Economie: {economie:.2f} lei")
#
# plt.figure(figsize=(5, 4))
# categorii = ['Cost inițial\n(medie populație)', 'Cost optim\n(individ final)', 'Economie']
# valori = [cost_total_inainte, cost_total_dupa, economie]
# culori = ['red', 'blue', 'green']
# bars = plt.bar(categorii, valori, color=culori, alpha=0.7, width=0.4, edgecolor='gray')
# for bar, val in zip(bars, valori):
#     plt.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.2f} lei', ha='center', va='bottom', fontsize=10)
# val_max77 = int(np.max(cost_total_inainte))
# plt.ylim(0, val_max77 + 20)
# plt.ylabel('Cost total zilnic (lei)')
# plt.title('Comparație cost energetic zilnic')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# #Apelare functii de afisare
# # 1. Grafic fitness minim
# afisare_fitness_minim(arhiva_fitness_minim, G_MAX)
# # 2. Grafic fitness mediu
# afisare_fitness_mediu(arhiva_fitness_mediu, G_MAX)
# # 3. GRAFIC CONSTRANGERI DE CONFORT INCALCATE PE ORA
# afisare_constrangeri_confort(individ_minim, Conf, N, dispozitive)
# # 4. PLANIFICAREA FUNCTIONALITATII INDIVIDULUI MINIM - DIAGRAMA GANTT
# afisare_planificare_gantt(baseline, TP, dispozitive, N, "Programarea baseline a dispozitivelor")
# afisare_planificare_gantt(individ_minim, TP, dispozitive, N, "Programarea optimă a dispozitivelor", fitness_minim)
# # 5. Graficul de consum
# CT_baseline = calcul_CT(baseline, PN, N)
# CT_optim = calcul_CT(individ_minim, PN, N)
#
# afisare_consum_orar(CT_baseline, CT_optim, 'Individ initial', 'Individ optim', 'Comparație profiluri de consum')
# # 6. Curba de consum/energie — profilul CT(h) × cost(h)
# cost_orar_baseline, cost_orar_optim = afisare_cost_orar(CT_baseline, CT_optim, cost,'Cost initial', 'Cost optim','Cost energie')
# # 7. GRAFIC EXPRIMARE COST TOTAL ININTE SI DUPA
# afisare_cost_total(cost_orar_baseline, cost_orar_optim,'Initial', 'Optim','Comparație cost energetic zilnic')

#sdfsdf
#sdfdsf