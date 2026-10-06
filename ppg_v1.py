#!/usr/bin/env python3

"""
Dragan Stosic
https://github.com/gajaka/ppg-lean

PPG znači Proof-Preserving Graphs — grafovi koji čuvaju dokazana svojstva pri promenama sistema.
U originalnom projektu služe da utvrde:
- koje uslove sistem zadovoljava;
- koji uslovi ga blokiraju;
- da li promena rešava problem i čuva potrebna svojstva.

U mom kodu oznaka „PPG-inspired“ znači da sam preuzeo ideju provera i blokirajućih uslova. 
Nisam implementirao originalni PPG niti njegove Lean dokaze. 
Predikciju NEXT daju modeli učestalosti i prelaza; PPG deo samo prikazuje rezultate provere.



PPG-inspired NEXT predictor for Loto 7/39.

Procedure:
1. Validate historical draws.
2. Predict each historical draw using only earlier draws.
3. Select one candidate using the validation period.
4. Evaluate that frozen selection on a separate test period.
5. Update its historical statistics and predict NEXT.

PPG-inspired assessment:
- explicit checks;
- inspectable blocking conditions;
- distinction between producing a prediction and passing assessment.

No Lean certification is claimed.
"""

import argparse
import csv
import math
import random
import statistics
from collections import Counter

N = 39
K = 7
BASE = K / N
RANDOM_HITS = K * BASE


def read_draws(path, reverse=False):
    draws = []

    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or all(not cell.strip() for cell in row):
                continue

            try:
                numbers = tuple(int(cell.strip()) for cell in row)
            except ValueError as exc:
                raise ValueError(
                    f"Red {line}: očekivano je 7 celih brojeva bez zaglavlja."
                ) from exc

            if len(numbers) != K:
                raise ValueError(f"Red {line}: ima {len(numbers)} brojeva.")
            if len(set(numbers)) != K:
                raise ValueError(f"Red {line}: brojevi se ponavljaju.")
            if not all(1 <= number <= N for number in numbers):
                raise ValueError(f"Red {line}: broj izvan opsega 1–39.")

            # Sorted CSV columns are treated as a set, not drawing order.
            draws.append(tuple(sorted(number - 1 for number in numbers)))

    if reverse:
        draws.reverse()

    return draws


class History:
    """Statistics containing only already observed draws."""

    def __init__(self):
        self.draws = []
        self.prefix = [[0] * N]

        # Directed counts: number i in one draw -> number j in next draw.
        self.transitions = [[0] * N for _ in range(N)]
        self.sources = [0] * N

    def observe(self, draw):
        if self.draws:
            previous = self.draws[-1]
            for source in previous:
                self.sources[source] += 1
                for target in draw:
                    self.transitions[source][target] += 1

        counts = self.prefix[-1].copy()
        for number in draw:
            counts[number] += 1

        self.prefix.append(counts)
        self.draws.append(draw)

    def frequency(self, window=None, strength=39.0):
        size = len(self.draws)
        start = 0 if window is None else max(0, size - window)
        count = size - start

        # Shrink observed frequencies toward 7/39.
        return [
            (
                self.prefix[size][number]
                - self.prefix[start][number]
                + strength * BASE
            ) / (count + strength)
            for number in range(N)
        ]

    def transition(self, strength=100.0):
        global_frequency = self.frequency()

        if not self.draws:
            return global_frequency

        previous = self.draws[-1]
        result = []

        for target in range(N):
            conditional = [
                (
                    self.transitions[source][target]
                    + strength * global_frequency[target]
                ) / (self.sources[source] + strength)
                for source in previous
            ]
            result.append(sum(conditional) / K)

        return result

    def scores(self):
        global_frequency = self.frequency()
        recent_100 = self.frequency(100)
        transition_100 = self.transition(100)

        result = {
            "frequency_all": global_frequency,
            "transition_25": self.transition(25),
            "transition_100": transition_100,
            "transition_400": self.transition(400),
            "blend_global_transition": [
                0.5 * global_frequency[i] + 0.5 * transition_100[i]
                for i in range(N)
            ],
            "blend_recent_transition": [
                0.5 * recent_100[i] + 0.5 * transition_100[i]
                for i in range(N)
            ],
        }

        for window in (25, 50, 100, 250, 500, 1000):
            frequency = self.frequency(window)
            result[f"frequency_{window}"] = frequency

            # A ranking heuristic, not a calibrated probability.
            result[f"cold_{window}"] = [
                2 * BASE - value for value in frequency
            ]

        return result


def choose_seven(scores, seed, step):
    """Reproducible random tie-breaking avoids favoring larger numbers."""
    rng = random.Random(seed + step * 1_000_003)
    ties = [rng.random() for _ in range(N)]
    ranking = sorted(
        range(N),
        key=lambda number: (scores[number], ties[number]),
        reverse=True,
    )
    return tuple(sorted(ranking[:K]))


def hits(prediction, actual):
    return len(set(prediction).intersection(actual))


def describe(values):
    size = len(values)
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if size > 1 else 0.0
    standard_error = deviation / math.sqrt(size)
    margin = 1.96 * standard_error

    return {
        "n": size,
        "mean": mean,
        "se": standard_error,
        "lower": mean - margin,
        "upper": mean + margin,
        "histogram": dict(sorted(Counter(values).items())),
    }


def fmt(draw):
    return " ".join(f"{number + 1:02d}" for number in draw)


def main():
    parser = argparse.ArgumentParser(
        description="History-based Loto 7/39 NEXT predictor."
    )
    # parser.add_argument("--csv", required=True)
    parser.add_argument(
    "--csv",
    default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
    # default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
    
    # default="/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
    # Potrebno je najmanje 500 početnih kola pre validacije. Smanji --validation i --test
    )
    
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--validation", type=int, default=1000) #500 1000
    parser.add_argument("--test", type=int, default=1000) #500 1000
    parser.add_argument("--seed", type=int, default=739)
    args = parser.parse_args()

    if args.validation < 100 or args.test < 100:
        parser.error("Validacija i test moraju imati najmanje 100 kola.")

    draws = read_draws(args.csv, args.reverse)
    count = len(draws)
    validation_start = count - args.validation - args.test
    test_start = count - args.test

    if validation_start < 500:
        parser.error(
            "Potrebno je najmanje 500 početnih kola pre validacije. "
            "Smanji --validation i --test ili koristi više podataka."
        )

    history = History()
    validation_hits = {}
    test_hits = {}
    selected = None
    example_test_predictions = []

    for step, actual in enumerate(draws):
        if step >= validation_start:
            # Freeze model choice before observing any test outcome.
            if step == test_start:
                selected = max(
                    validation_hits,
                    key=lambda name: (
                        statistics.mean(validation_hits[name]),
                        name,
                    ),
                )

            scores = history.scores()

            if step < test_start:
                for name, ranking_scores in scores.items():
                    prediction = choose_seven(
                        ranking_scores, args.seed, step
                    )
                    validation_hits.setdefault(name, []).append(
                        hits(prediction, actual)
                    )
            else:
                prediction = choose_seven(
                    scores[selected], args.seed, step
                )
                result = hits(prediction, actual)
                test_hits.setdefault(selected, []).append(result)

                if step >= count - 5:
                    example_test_predictions.append(
                        (step + 1, prediction, actual, result)
                    )

        # Reveal current outcome only after making its prediction.
        history.observe(actual)

    validation = describe(validation_hits[selected])
    test = describe(test_hits[selected])

    next_scores = history.scores()[selected]
    next_prediction = choose_seven(next_scores, args.seed, count)

    # Baseline variability under independent uniform 7/39 draws.
    null_variance = (
        K * BASE * (1 - BASE) * (N - K) / (N - 1)
    )
    null_se = math.sqrt(null_variance / args.test)
    null_z = (test["mean"] - RANDOM_HITS) / null_se

    # Assessment is conservative: a NEXT output can exist while the
    # empirical predictive-advantage check remains blocked.
    checks = [
        (
            "CSV: sedam različitih brojeva u opsegu 1–39",
            True,
        ),
        (
            "Validacija i test su odvojeni",
            True,
        ),
        (
            "Izbor modela završen pre test perioda",
            True,
        ),
        (
            "Svaka istorijska predikcija koristi samo prethodna kola",
            True,
        ),
        (
            "NEXT: sedam različitih brojeva u opsegu 1–39",
            len(next_prediction) == K
            and len(set(next_prediction)) == K
            and all(0 <= number < N for number in next_prediction),
        ),
        (
            "Donja granica približnog test intervala iznad "
            "slučajnog očekivanja",
            test["lower"] > RANDOM_HITS,
        ),
    ]

    print(f"\nUčitano izvlačenja: {count}")
    print("Pretpostavka: redovi su od najstarijeg ka najnovijem.")
    print(f"Poslednje izvlačenje: {fmt(draws[-1])}")
    print(f"Početna istorija: {validation_start} kola")
    print(
        f"Validacija: redovi {validation_start + 1}–{test_start}"
    )
    print(f"Test: redovi {test_start + 1}–{count}")

    print("\nMODELI — rezultat samo na validaciji")
    for name in sorted(
        validation_hits,
        key=lambda item: statistics.mean(validation_hits[item]),
        reverse=True,
    ):
        mean = statistics.mean(validation_hits[name])
        marker = " <-- izabran" if name == selected else ""
        print(f"{name:30s} {mean:.4f}{marker}")

    print(f"\nIZABRANI MODEL: {selected}")
    print(f"Validacija, pogodaka po kolu: {validation['mean']:.4f}")
    print(f"Test, pogodaka po kolu:       {test['mean']:.4f}")
    print(f"Slučajno očekivanje:         {RANDOM_HITS:.4f}")
    print(
        "Približan 95% interval test proseka: "
        f"[{test['lower']:.4f}, {test['upper']:.4f}]"
    )
    print(f"Odstupanje od uniformnog modela, z: {null_z:.3f}")
    print(
        "Interval je opisna normalna aproksimacija; njegova "
        "interpretacija zahteva odgovarajuću nezavisnost ishoda."
    )

    print("\nPOGOCI NA TESTU")
    for number in range(K + 1):
        print(f"{number}: {test['histogram'].get(number, 0)} kola")

    print("\nPOSLEDNJIH PET ISTORIJSKIH PREDIKCIJA")
    for row, prediction, actual, result in example_test_predictions:
        print(
            f"Red {row}: pred {fmt(prediction)} | "
            f"stvarno {fmt(actual)} | pogodaka {result}"
        )

    print("\nPPG-INSPIRED PROVERA")
    for label, passed in checks:
        print(f"[{'PROLAZI' if passed else 'BLOKIRANO'}] {label}")

    blocked = [label for label, passed in checks if not passed]
    if blocked:
        print(
            "\nStatus: NEXT je izračunat, ali prediktivna prednost "
            "nije potvrđena ovom proverom."
        )
    else:
        print(
            "\nStatus: prediktor prolazi navedene empirijske kriterijume. "
            "To nije garancija budućeg rezultata."
        )

    print(f"\nNEXT: {fmt(next_prediction)}")
    print(
        "NEXT bira sedam najviše rangiranih brojeva. "
        "Optimizuje modelsku procenu broja pogodaka; "
        "ne računa zajedničku verovatnoću tačne sedmorke."
    )

    print("\nRANGIRANJE BROJEVA ZA NEXT — skorovi nisu verovatnoće")
    ranking = sorted(
        range(N),
        key=lambda number: next_scores[number],
        reverse=True,
    )
    for position, number in enumerate(ranking, 1):
        marker = "*" if number in next_prediction else " "
        print(
            f"{position:2d}. {number + 1:02d} "
            f"skor={next_scores[number]:.6f} {marker}"
        )


if __name__ == "__main__":
    main()



"""
Učitano izvlačenja: 4696
Pretpostavka: redovi su od najstarijeg ka najnovijem.
Poslednje izvlačenje: 05 06 16 21 23 26 36
Početna istorija: 2696 kola
Validacija: redovi 2697–3696
Test: redovi 3697–4696

MODELI — rezultat samo na validaciji
cold_100                       1.3110 <-- izabran
transition_100                 1.3040
transition_25                  1.2940
blend_recent_transition        1.2900
cold_500                       1.2880
cold_1000                      1.2800
cold_50                        1.2710
cold_250                       1.2670
transition_400                 1.2640
blend_global_transition        1.2620
frequency_100                  1.2610
frequency_25                   1.2600
frequency_1000                 1.2590
frequency_50                   1.2420
frequency_all                  1.2340
cold_25                        1.2270
frequency_250                  1.2020
frequency_500                  1.2000

IZABRANI MODEL: cold_100
Validacija, pogodaka po kolu: 1.3110
Test, pogodaka po kolu:       1.2660
Slučajno očekivanje:         1.2564
Približan 95% interval test proseka: [1.2081, 1.3239]
Odstupanje od uniformnog modela, z: 0.325
Interval je opisna normalna aproksimacija; njegova interpretacija zahteva odgovarajuću nezavisnost ishoda.

POGOCI NA TESTU
0: 215 kola
1: 410 kola
2: 286 kola
3: 73 kola
4: 15 kola
5: 1 kola
6: 0 kola
7: 0 kola

POSLEDNJIH PET ISTORIJSKIH PREDIKCIJA
Red 4692: pred 13 21 23 26 32 36 37 | stvarno 03 05 12 14 15 23 31 | pogodaka 1
Red 4693: pred 13 21 23 26 32 36 37 | stvarno 01 07 10 21 29 32 36 | pogodaka 3
Red 4694: pred 13 21 23 26 32 36 37 | stvarno 08 12 19 20 27 34 38 | pogodaka 0
Red 4695: pred 13 21 23 26 32 36 37 | stvarno 04 10 14 15 28 33 34 | pogodaka 0
Red 4696: pred 13 21 23 26 32 36 37 | stvarno 05 06 16 21 23 26 36 | pogodaka 4

PPG-INSPIRED PROVERA
[PROLAZI] CSV: sedam različitih brojeva u opsegu 1–39
[PROLAZI] Validacija i test su odvojeni
[PROLAZI] Izbor modela završen pre test perioda
[PROLAZI] Svaka istorijska predikcija koristi samo prethodna kola
[PROLAZI] NEXT: sedam različitih brojeva u opsegu 1–39
[BLOKIRANO] Donja granica približnog test intervala iznad slučajnog očekivanja

Status: NEXT je izračunat, ali prediktivna prednost nije potvrđena ovom proverom.

NEXT: 13 21 23 26 32 36 37
NEXT bira sedam najviše rangiranih brojeva. Optimizuje modelsku procenu broja pogodaka; ne računa zajedničku verovatnoću tačne sedmorke.

RANGIRANJE BROJEVA ZA NEXT — skorovi nisu verovatnoće
 1. 26 skor=0.243866 *
 2. 37 skor=0.236672 *
 3. 21 skor=0.222284 *
 4. 32 skor=0.222284 *
 5. 13 skor=0.215089 *
 6. 23 skor=0.207895 *
 7. 35 skor=0.207895  
 8. 36 skor=0.207895 *
 9. 39 skor=0.207895  
10. 02 skor=0.193507  
11. 11 skor=0.193507  
12. 33 skor=0.193507  
13. 04 skor=0.186312  
14. 06 skor=0.186312  
15. 09 skor=0.186312  
16. 20 skor=0.186312  
17. 25 skor=0.186312  
18. 27 skor=0.186312  
19. 03 skor=0.179118  
20. 15 skor=0.179118  
21. 17 skor=0.179118  
22. 22 skor=0.179118  
23. 24 skor=0.179118  
24. 34 skor=0.179118  
25. 30 skor=0.171924  
26. 18 skor=0.164730  
27. 28 skor=0.164730  
28. 29 skor=0.164730  
29. 38 skor=0.164730  
30. 05 skor=0.157536  
31. 08 skor=0.157536  
32. 16 skor=0.157536  
33. 12 skor=0.150341  
34. 19 skor=0.150341  
35. 31 skor=0.150341  
36. 10 skor=0.143147  
37. 07 skor=0.121564  
38. 14 skor=0.121564  
39. 01 skor=0.114370





Učitano izvlačenja: 2970
Pretpostavka: redovi su od najstarijeg ka najnovijem.
Poslednje izvlačenje: 04 10 14 15 28 33 34
Početna istorija: 970 kola
Validacija: redovi 971–1970
Test: redovi 1971–2970

MODELI — rezultat samo na validaciji
cold_50                        1.3120 <-- izabran
frequency_500                  1.2970
cold_25                        1.2960
cold_100                       1.2800
cold_250                       1.2800
frequency_all                  1.2720
cold_500                       1.2720
cold_1000                      1.2660
transition_100                 1.2650
transition_25                  1.2630
transition_400                 1.2630
blend_global_transition        1.2620
frequency_1000                 1.2540
frequency_250                  1.2230
blend_recent_transition        1.2090
frequency_50                   1.2040
frequency_25                   1.1960
frequency_100                  1.1860

IZABRANI MODEL: cold_50
Validacija, pogodaka po kolu: 1.3120
Test, pogodaka po kolu:       1.2490
Slučajno očekivanje:         1.2564
Približan 95% interval test proseka: [1.1925, 1.3055]
Odstupanje od uniformnog modela, z: -0.252
Interval je opisna normalna aproksimacija; njegova interpretacija zahteva odgovarajuću nezavisnost ishoda.

POGOCI NA TESTU
0: 207 kola
1: 440 kola
2: 262 kola
3: 79 kola
4: 12 kola
5: 0 kola
6: 0 kola
7: 0 kola

POSLEDNJIH PET ISTORIJSKIH PREDIKCIJA
Red 2966: pred 02 04 13 21 23 25 34 | stvarno 01 10 15 19 23 25 39 | pogodaka 2
Red 2967: pred 02 04 11 13 21 25 34 | stvarno 09 12 14 16 21 22 36 | pogodaka 1
Red 2968: pred 02 04 13 21 25 26 34 | stvarno 01 05 09 17 19 27 34 | pogodaka 1
Red 2969: pred 02 04 13 21 23 25 26 | stvarno 01 07 10 21 29 32 36 | pogodaka 1
Red 2970: pred 02 04 13 21 25 26 34 | stvarno 04 10 14 15 28 33 34 | pogodaka 2

PPG-INSPIRED PROVERA
[PROLAZI] CSV: sedam različitih brojeva u opsegu 1–39
[PROLAZI] Validacija i test su odvojeni
[PROLAZI] Izbor modela završen pre test perioda
[PROLAZI] Svaka istorijska predikcija koristi samo prethodna kola
[PROLAZI] NEXT: sedam različitih brojeva u opsegu 1–39
[BLOKIRANO] Donja granica približnog test intervala iznad slučajnog očekivanja

Status: NEXT je izračunat, ali prediktivna prednost nije potvrđena ovom proverom.

NEXT: 02 04 13 21 25 26 35
NEXT bira sedam najviše rangiranih brojeva. Optimizuje modelsku procenu broja pogodaka; ne računa zajedničku verovatnoću tačne sedmorke.

RANGIRANJE BROJEVA ZA NEXT — skorovi nisu verovatnoće
 1. 02 skor=0.235379 *
 2. 04 skor=0.224143 *
 3. 13 skor=0.224143 *
 4. 21 skor=0.224143 *
 5. 25 skor=0.224143 *
 6. 26 skor=0.224143 *
 7. 35 skor=0.224143 *
 8. 11 skor=0.212907  
 9. 23 skor=0.212907  
10. 37 skor=0.212907  
11. 17 skor=0.201671  
12. 27 skor=0.201671  
13. 33 skor=0.201671  
14. 34 skor=0.201671  
15. 03 skor=0.190435  
16. 09 skor=0.190435  
17. 20 skor=0.190435  
18. 24 skor=0.190435  
19. 31 skor=0.190435  
20. 32 skor=0.190435  
21. 15 skor=0.179199  
22. 19 skor=0.179199  
23. 36 skor=0.179199  
24. 06 skor=0.167963  
25. 08 skor=0.167963  
26. 12 skor=0.167963  
27. 18 skor=0.167963  
28. 38 skor=0.167963  
29. 39 skor=0.156727  
30. 05 skor=0.145491  
31. 16 skor=0.145491  
32. 22 skor=0.145491  
33. 29 skor=0.145491  
34. 14 skor=0.134255  
35. 30 skor=0.134255  
36. 01 skor=0.111783  
37. 07 skor=0.111783  
38. 10 skor=0.111783  
39. 28 skor=0.111783  





Učitano izvlačenja: 1726
Pretpostavka: redovi su od najstarijeg ka najnovijem.
Poslednje izvlačenje: 05 06 16 21 23 26 36
Početna istorija: 726 kola
Validacija: redovi 727–1226
Test: redovi 1227–1726

MODELI — rezultat samo na validaciji
blend_global_transition        1.3160 <-- izabran
transition_400                 1.3140
cold_250                       1.3140
frequency_all                  1.3060
frequency_25                   1.2980
frequency_1000                 1.2980
cold_500                       1.2940
transition_100                 1.2720
frequency_250                  1.2720
cold_50                        1.2700
transition_25                  1.2660
cold_1000                      1.2660
frequency_50                   1.2580
frequency_500                  1.2540
cold_25                        1.2480
cold_100                       1.2460
frequency_100                  1.2400
blend_recent_transition        1.2380

IZABRANI MODEL: blend_global_transition
Validacija, pogodaka po kolu: 1.3160
Test, pogodaka po kolu:       1.2900
Slučajno očekivanje:         1.2564
Približan 95% interval test proseka: [1.2065, 1.3735]
Odstupanje od uniformnog modela, z: 0.806
Interval je opisna normalna aproksimacija; njegova interpretacija zahteva odgovarajuću nezavisnost ishoda.

POGOCI NA TESTU
0: 105 kola
1: 204 kola
2: 141 kola
3: 43 kola
4: 5 kola
5: 2 kola
6: 0 kola
7: 0 kola

POSLEDNJIH PET ISTORIJSKIH PREDIKCIJA
Red 1722: pred 02 08 11 23 26 29 37 | stvarno 02 05 06 10 12 22 30 | pogodaka 1
Red 1723: pred 02 07 11 23 26 34 37 | stvarno 03 08 12 17 18 30 33 | pogodaka 0
Red 1724: pred 02 07 08 11 23 26 29 | stvarno 03 05 12 14 15 23 31 | pogodaka 1
Red 1725: pred 02 07 11 18 23 26 34 | stvarno 08 12 19 20 27 34 38 | pogodaka 1
Red 1726: pred 02 08 11 23 26 29 34 | stvarno 05 06 16 21 23 26 36 | pogodaka 2

PPG-INSPIRED PROVERA
[PROLAZI] CSV: sedam različitih brojeva u opsegu 1–39
[PROLAZI] Validacija i test su odvojeni
[PROLAZI] Izbor modela završen pre test perioda
[PROLAZI] Svaka istorijska predikcija koristi samo prethodna kola
[PROLAZI] NEXT: sedam različitih brojeva u opsegu 1–39
[BLOKIRANO] Donja granica približnog test intervala iznad slučajnog očekivanja

Status: NEXT je izračunat, ali prediktivna prednost nije potvrđena ovom proverom.

NEXT: 02 11 23 26 32 34 37
NEXT bira sedam najviše rangiranih brojeva. Optimizuje modelsku procenu broja pogodaka; ne računa zajedničku verovatnoću tačne sedmorke.

RANGIRANJE BROJEVA ZA NEXT — skorovi nisu verovatnoće
 1. 23 skor=0.201585 *
 2. 34 skor=0.196316 *
 3. 02 skor=0.195771 *
 4. 11 skor=0.195647 *
 5. 26 skor=0.191568 *
 6. 37 skor=0.190786 *
 7. 32 skor=0.188268 *
 8. 08 skor=0.188212  
 9. 04 skor=0.187434  
10. 10 skor=0.187074  
11. 07 skor=0.185303  
12. 18 skor=0.185202  
13. 31 skor=0.184009  
14. 29 skor=0.183326  
15. 27 skor=0.183212  
16. 25 skor=0.183191  
17. 12 skor=0.179828  
18. 33 skor=0.179685  
19. 38 skor=0.178588  
20. 09 skor=0.177075  
21. 39 skor=0.176675  
22. 14 skor=0.176571  
23. 05 skor=0.176357  
24. 13 skor=0.176275  
25. 28 skor=0.175450  
26. 35 skor=0.175408  
27. 24 skor=0.173938  
28. 36 skor=0.173634  
29. 03 skor=0.173599  
30. 01 skor=0.173412  
31. 19 skor=0.173253  
32. 22 skor=0.171479  
33. 20 skor=0.169133  
34. 06 skor=0.169085  
35. 30 skor=0.167955  
36. 16 skor=0.166305  
37. 15 skor=0.165809  
38. 21 skor=0.163912  
39. 17 skor=0.159672 
"""
