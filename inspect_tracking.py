import matplotlib.pyplot as plt
from statistics import mean, median


total_frames = 0
valid_frames = []
missing_frames = []

with open("data/tactical/BAR-ATH_tactical_atd.txt", "r") as f:
    for line in f:
        frame, players_text, ball_text = line.strip().split(":")
        x, y, s = ball_text.split(",")

        total_frames += 1

        if x != "NaN" and y != "NaN":
            valid_frames.append(int(frame))
        else:
            missing_frames.append(int(frame))

print("Total cadre:", total_frames)
print("Cadre cu poziția mingii:", len(valid_frames))
print("Cadre fără poziția mingii:", len(missing_frames))

print(
    "Procent cadre valide:",
    round(len(valid_frames) / total_frames * 100, 2),
    "%"
)

if valid_frames:
    print("Primul cadru valid:", valid_frames[0])
    print("Ultimul cadru valid:", valid_frames[-1])


intervale_lipsa = []

if missing_frames:
    start = missing_frames[0]
    anterior = start

    for frame in missing_frames[1:]:
        if frame != anterior + 1:
            intervale_lipsa.append((start, anterior))
            start = frame

        anterior = frame

    intervale_lipsa.append((start, anterior))

fps = 25

durate = [
    (sfarsit - inceput + 1) / fps
    for inceput, sfarsit in intervale_lipsa
]

print("\n--- Intervale fara minge ---")
print("Numar intervale:", len(intervale_lipsa))

if durate:
    print("Durata medie:", round(mean(durate), 3), "secunde")
    print("Durata mediana:", round(median(durate), 3), "secunde")
    print("Durata maxima:", round(max(durate), 3), "secunde")


# Construim toate intervalele: cu minge si fara minge
toate_intervalele = []

start = 1
stare_anterioara = 1 if 1 in set(valid_frames) else 0

valid_frames_set = set(valid_frames)

for frame in range(2, total_frames + 1):
    stare_curenta = 1 if frame in valid_frames_set else 0

    if stare_curenta != stare_anterioara:
        toate_intervalele.append(
            (start, frame - 1, stare_anterioara)
        )

        start = frame
        stare_anterioara = stare_curenta

# Salvam si ultimul interval
toate_intervalele.append(
    (start, total_frames, stare_anterioara)
)

print("Total intervale:", len(toate_intervalele))

print("Primele 5 intervale:", toate_intervalele[:5])

# Analizam disponibilitatea mingii in fiecare repriza
reprize = [
    (1, 11186, 90652),
    (2, 110809, 190198)
]

for nr, start, end in reprize:
    total = end - start + 1

    cadre_cu_minge = sum(
        1 for frame in range(start, end + 1)
        if frame in valid_frames_set
    )

    procent = cadre_cu_minge / total * 100

    print(f"\nRepriza {nr}:")
    print("Total cadre:", total)
    print("Cadre cu minge:", cadre_cu_minge)
    print("Cadre fara minge:", total - cadre_cu_minge)
    print("Procent disponibilitate:", round(procent, 2), "%")


import csv

with open("ball_intervals.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Interval",
        "Cadru inceput",
        "Cadru sfarsit",
        "Stim pozitia mingii",
        "Durata (secunde)"
    ])

    for i, (start, end, stare) in enumerate(toate_intervalele, start=1):
        durata = (end - start + 1) / fps

        writer.writerow([
            i,
            start,
            end,
            "DA" if stare == 1 else "NU",
            round(durata, 2)
        ])

print("Tabelul ball_intervals.csv a fost creat!")


# # Construim semnalul binar: 1 = avem mingea, 0 = lipsește
# times = []
# signal = []

# valid_frames_set = set(valid_frames)

# for frame in range(1, total_frames + 1):
#     times.append((frame - 1) / fps / 60)

#     if frame in valid_frames_set:
#         signal.append(1)
#     else:
#         signal.append(0)

# plt.figure(figsize=(15, 4))

# plt.step(times, signal, where="post", color="blue", linewidth=1)

# plt.xlabel("Timp (minute)")
# plt.ylabel("Disponibilitatea mingii")
# plt.yticks([0, 1], ["0 = Nu", "1 = Da"])
# plt.ylim(-0.2, 1.2)
# plt.title("Semnal binar - disponibilitatea mingii")
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()