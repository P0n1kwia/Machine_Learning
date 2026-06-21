# CS229 — Moduł 1: Regresja liniowa — Zestaw zadań (część matematyczna)

> **Zakres:** Rozdział 1 `main_notes` (LMS / gradient descent, równania normalne,
> interpretacja probabilistyczna = MLE, LWR). **Bez kodu** — część programistyczną
> masz w osobnym notebooku.
>
> **Oznaczenia:** 🟦 rdzeń (rozwiązywalne wprost materiałem rozdziału) · 🟧 stretch
> (rozszerzenie / synteza / przypadek brzegowy). Wskazówki są opcjonalne — najpierw spróbuj bez.
>
> **Notacja.** Macierz projektowa $X\in\mathbb{R}^{n\times d}$ ma przykłady w wierszach,
> $x^{(i)}\in\mathbb{R}^{d}$ z konwencją $x_0=1$ (intercept). $\vec{y}\in\mathbb{R}^n$,
> hipoteza $h_\theta(x)=\theta^T x$, koszt $J(\theta)=\tfrac12\sum_{i=1}^n\big(h_\theta(x^{(i)})-y^{(i)}\big)^2$.
>
> ⚠️ **Uczciwość:** przy długich wyprowadzeniach mogłem coś przeoczyć — kluczowe kroki
> weryfikuj z `main_notes.pdf` (Rozdz. 1) i z sekcją „Linear Algebra Review” z linków w planie.

---

## 0. Powtórka / rozgrzewka (prerekwizyty)

To moduł 1, więc nie ma jeszcze wcześniejszego materiału kursu do powtórki. Zamiast tego —
dwie krótkie rozgrzewki z narzędzi, których zaraz użyjemy.

**0.1 🟦 (rachunek macierzowy).** Niech $A\in\mathbb{R}^{2\times2}$ będzie **symetryczna**
($A_{12}=A_{21}$) i $x=(x_1,x_2)^T$. Rozpisz $f(x)=x^TAx$ jawnie jako wielomian w $x_1,x_2$,
policz $\partial f/\partial x_1$ oraz $\partial f/\partial x_2$ i pokaż, że
$$\nabla_x\,(x^TAx)=2Ax.$$
Gdzie w wyprowadzeniu używasz symetrii $A$? Co się psuje, gdy $A$ nie jest symetryczna
(podaj poprawny wzór ogólny)?

**0.2 🟦 (prawdopodobieństwo / MLE).** Zapisz gęstość rozkładu $\mathcal{N}(\mu,\sigma^2)$.
Dla próbki $z_1,\dots,z_n$ i.i.d. z $\mathcal{N}(\mu,\sigma^2)$ wypisz log-likelihood $\ell(\mu)$,
a następnie znajdź estymator największej wiarogodności (MLE) $\hat\mu$. Pokaż po drodze, że
maksymalizacja $\ell$ względem $\mu$ jest równoważna **minimalizacji** $\sum_i (z_i-\mu)^2$ —
to dokładnie ten sam ruch, którego użyjemy w zad. 1.4.

---

## 1. Zadania rdzeniowe

**1.1 🟦 (pochodna kosztu → reguła LMS).** Dla **jednego** przykładu $(x,y)$ pokaż krok po kroku, że
$$\frac{\partial}{\partial\theta_j}\,\tfrac12\big(h_\theta(x)-y\big)^2=\big(h_\theta(x)-y\big)\,x_j,$$
i wyprowadź stąd regułę aktualizacji LMS (Widrow–Hoff):
$\theta_j:=\theta_j+\alpha\big(y-h_\theta(x)\big)x_j$. Wyjaśnij intuicję: dlaczego wielkość
poprawki jest proporcjonalna do błędu predykcji?

**1.2 🟦 (gradient w postaci wektorowej + batch GD).** Korzystając z 1.1, pokaż, że dla całego
zbioru
$$\nabla_\theta J(\theta)=\sum_{i=1}^n\big(h_\theta(x^{(i)})-y^{(i)}\big)x^{(i)}=X^T(X\theta-\vec{y}),$$
i zapisz aktualizację **batch gradient descent** jako $\theta:=\theta-\alpha\nabla_\theta J(\theta)$.
Sprawdź zgodność wymiarów każdego czynnika w $X^T(X\theta-\vec{y})$.

**1.3 🟦 ⭐ (pełne wyprowadzenie — równania normalne).** Wychodząc od
$J(\theta)=\tfrac12(X\theta-\vec{y})^T(X\theta-\vec{y})$, wyprowadź
$$\nabla_\theta J(\theta)=X^TX\theta-X^T\vec{y},$$
**uzasadniając każdy krok**. W szczególności jawnie zaznacz, gdzie używasz:
(a) $a^Tb=b^Ta$ dla wektorów, oraz (b) tożsamości $\nabla_x\,b^Tx=b$ i
$\nabla_x\,x^TAx=2Ax$ dla symetrycznej $A$ (tu $A=X^TX$ — uzasadnij, że jest symetryczna).
Następnie przyrównaj gradient do zera, otrzymaj **równanie normalne** $X^TX\theta=X^T\vec{y}$
i wzór zamknięty $\theta=(X^TX)^{-1}X^T\vec{y}$.
*To jest „wymagane pełne wyprowadzenie” z tego zestawu.*

**1.4 🟦 (interpretacja probabilistyczna = MLE).** Przyjmij model
$y^{(i)}=\theta^T x^{(i)}+\varepsilon^{(i)}$, gdzie $\varepsilon^{(i)}\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2)$.
Wypisz wiarogodność $L(\theta)$, przejdź do log-likelihood $\ell(\theta)$ i pokaż, że
$$\ell(\theta)=n\log\frac{1}{\sqrt{2\pi}\,\sigma}-\frac{1}{\sigma^2}\cdot\tfrac12\sum_{i=1}^n\big(y^{(i)}-\theta^T x^{(i)}\big)^2.$$
Wywnioskuj, że maksymalizacja $\ell(\theta)$ jest równoważna minimalizacji $J(\theta)$.
**Dlaczego** wynik (czyli $\hat\theta$) nie zależy od $\sigma^2$? (Tę obserwację wykorzystamy
później przy rodzinie wykładniczej i GLM.)

**1.5 🟦 (zrozumienie — batch vs SGD).** W kilku zdaniach porównaj **batch** i **stochastic**
gradient descent: koszt jednego kroku (jako funkcja $n$), tempo wczesnego postępu, zachowanie
w pobliżu minimum. Co dokładnie znaczy, że SGD „oscyluje” wokół minimum $J(\theta)$ i dlaczego
harmonogram $\alpha_t\to 0$ (np. $\alpha_t=\alpha_0/(1+kt)$) pozwala temu zaradzić? Kiedy wybrałbyś
SGD, a kiedy równanie normalne?

**1.6 🟦 (LWR — definicja i rola $\tau$).** Wypisz ważoną funkcję kosztu LWR
$\sum_i w^{(i)}(y^{(i)}-\theta^T x^{(i)})^2$ oraz standardowy wybór wag
$w^{(i)}=\exp\!\big(-\tfrac{(x^{(i)}-x)^2}{2\tau^2}\big)$. Opisz, co się dzieje z dopasowaniem
w granicach $\tau\to 0^+$ oraz $\tau\to\infty$. Wyjaśnij, dlaczego LWR jest algorytmem
**nieparametrycznym** (czym to się różni od „parametrycznej” regresji liniowej w sensie tego,
co musimy przechowywać, by robić predykcje?).

---

## 2. Zadania stretch (rozszerzenia i syntezy)

**2.1 🟧 ⭐ (synteza: ważone równania normalne).** Niech $W=\mathrm{diag}(w^{(1)},\dots,w^{(n)})$,
$w^{(i)}\ge 0$. Pokaż, że ważony koszt można zapisać jako
$$\tfrac12\sum_i w^{(i)}\big(\theta^T x^{(i)}-y^{(i)}\big)^2=\tfrac12(X\theta-\vec{y})^T W(X\theta-\vec{y}),$$
a następnie powtórz wyprowadzenie z 1.3, by otrzymać **ważone równanie normalne**
$$\theta=(X^TWX)^{-1}X^TW\vec{y}.$$
To jest dokładnie ten wzór, który zaimplementujesz w notebooku dla LWR (osobny $\theta$ na każde
zapytanie $x$). Jaki warunek na $W$ i $X$ gwarantuje odwracalność $X^TWX$?

**2.2 🟧 (wypukłość i jednoznaczność).** Policz Hesjan $\nabla^2_\theta J(\theta)$ i pokaż, że
$\nabla^2_\theta J=X^TX\succeq 0$ (półdodatnio określony), więc $J$ jest **wypukła**. Podaj warunek,
przy którym $J$ jest **ściśle** wypukła (i $X^TX\succ 0$). Powiąż to z: (a) jednoznacznością
rozwiązania równań normalnych, (b) faktem z rozdziału, że dla regresji liniowej GD zbiega do
minimum **globalnego** (brak minimów lokalnych).

**2.3 🟧 ⭐ (synteza: heteroskedastyczność → ważone najmniejsze kwadraty → LWR).**
Załóżmy teraz $\varepsilon^{(i)}\sim\mathcal{N}(0,\sigma_i^2)$ niezależne, ale o **różnych**
wariancjach (heteroscedastic). Wyprowadź MLE dla $\theta$ i pokaż, że prowadzi ono do
ważonych najmniejszych kwadratów z wagami $w^{(i)}=1/\sigma_i^2$. Zinterpretuj wagi LWR z zad. 1.6
przez ten pryzmat: w jakim sensie „lokalność” odpowiada przekonaniu, że obserwacje dalekie od
punktu zapytania są *de facto* „mniej wiarygodne” dla lokalnej predykcji? (To pomost do GLM i do
tematu wariancji błędu, który wróci później.)

**2.4 🟧 (geometria: rzut ortogonalny).** Oznacz $\hat{\vec{y}}=X\hat\theta=X(X^TX)^{-1}X^T\vec{y}$
i $H=X(X^TX)^{-1}X^T$ (tzw. *hat matrix*). Pokaż, że:
(a) $X^T(\vec{y}-\hat{\vec{y}})=0$ — residua są prostopadłe do kolumn $X$;
(b) $H^2=H$ oraz $H^T=H$ (czyli $H$ jest **rzutem ortogonalnym** na przestrzeń kolumn $X$);
(c) zinterpretuj równanie normalne jako warunek „residuum $\perp$ przestrzeń cech”. Jak to
geometrycznie tłumaczy, dlaczego najmniejsze kwadraty dają „najlepsze” przybliżenie $\vec{y}$
w $\mathrm{col}(X)$?

---

## Co dalej (oficjalny materiał)

- **Najpierw** zrób powyższe + notebook do modułu 1.
- **Oficjalny pset powiązany z tym modułem:** Problem Set 1, **Problem 5 — Locally Weighted
  Linear Regression** (LWR). Możesz go ruszyć **już teraz**, bo LWR jest w tym rozdziale; pełny
  PS1 (logistic/GDA/GLM/Poisson) zostaw na **po module 4** zgodnie z planem.
  - Repo: `maxim5/cs229-2018-autumn`
  - Zadanie: `problem-sets/PS1/PS1-5 Locally weighted linear regression.ipynb`
  - Kod startowy: `problem-sets/PS1/src/p05b_lwr.py` (oraz `p05c_tau.py` — dobór $\tau$)
  - Dane: `ds5_train.csv`, `ds5_valid.csv`, `ds5_test.csv`
  - Link: https://github.com/maxim5/cs229-2018-autumn/tree/main/problem-sets/PS1

---

<details>
<summary>📕 <b>Szkice rozwiązań — NIE ZAGLĄDAJ, dopóki nie spróbujesz sam.</b> (kliknij, by rozwinąć)</summary>

**0.1.** $f=A_{11}x_1^2+2A_{12}x_1x_2+A_{22}x_2^2$ (użyto $A_{12}=A_{21}$).
$\partial f/\partial x_1=2A_{11}x_1+2A_{12}x_2$, $\partial f/\partial x_2=2A_{12}x_1+2A_{22}x_2$,
co składa się w $2Ax$. Bez symetrii: $\nabla_x x^TAx=(A+A^T)x$ (redukuje się do $2Ax$, gdy $A=A^T$).

**0.2.** $p(z)=\tfrac{1}{\sqrt{2\pi}\sigma}e^{-(z-\mu)^2/(2\sigma^2)}$.
$\ell(\mu)=\text{const}-\tfrac{1}{2\sigma^2}\sum_i(z_i-\mu)^2$. Maksymalizacja po $\mu$ ⇔ minimalizacja
$\sum_i(z_i-\mu)^2$; pochodna = 0 daje $\hat\mu=\tfrac1n\sum_i z_i$.

**1.1.** Reguła łańcuchowa: $2\cdot\tfrac12(h-y)\cdot\partial_{\theta_j}(\sum_k\theta_k x_k-y)=(h-y)x_j$.
Stąd update $\theta_j:=\theta_j-\alpha(h-y)x_j=\theta_j+\alpha(y-h)x_j$. Poprawka $\propto$ błąd:
trafna predykcja → mała zmiana; duży błąd → duża zmiana.

**1.2.** Sumujesz 1.1 po $i$ i zbierasz $j$-tą współrzędną; w zwartej postaci wektor błędów to
$X\theta-\vec{y}$ (wymiar $n$), a $X^T(\cdot)$ daje wektor $d$. Wymiary: $X^T\,(n{\times}d)^T=d{\times}n$,
razy $n$-wektor → $d$-wektor. ✓

**1.3.** Rozwiń $(X\theta-\vec{y})^T(X\theta-\vec{y})=\theta^TX^TX\theta-2(X^T\vec{y})^T\theta+\vec{y}^T\vec{y}$
(środkowe wyrazy sklejasz przez $a^Tb=b^Ta$). $\nabla_\theta$: pierwszy wyraz → $2X^TX\theta$
(bo $X^TX$ symetr., $\nabla x^TAx=2Ax$), drugi → $-2X^T\vec{y}$ (bo $\nabla b^Tx=b$), trzeci → $0$.
Razy $\tfrac12$: $X^TX\theta-X^T\vec{y}$. Zero → $X^TX\theta=X^T\vec{y}$ → $\theta=(X^TX)^{-1}X^T\vec{y}$.
$(X^TX)^T=X^TX$ ⇒ symetria.

**1.4.** $L(\theta)=\prod_i\tfrac{1}{\sqrt{2\pi}\sigma}e^{-(y^{(i)}-\theta^Tx^{(i)})^2/(2\sigma^2)}$;
$\log$ zamienia iloczyn na sumę → podany wzór. Tylko ostatni człon zależy od $\theta$ i wchodzi z
minusem oraz dodatnim współczynnikiem $1/\sigma^2$, więc $\arg\max_\theta\ell=\arg\min_\theta J$.
$\sigma^2>0$ jest tylko stałym, dodatnim przeskalowaniem członu z $\theta$ → nie zmienia $\arg\min$,
stąd $\hat\theta$ niezależne od $\sigma^2$.

**1.5.** Batch: koszt kroku $O(n)$, stabilny, ale wolny gdy $n$ duże. SGD: koszt kroku $O(1)$,
szybki wczesny postęp, ale $\theta$ „skacze” wokół minimum, bo każdy przykład daje zaszumiony
estymator gradientu; malejące $\alpha_t$ wygasza te oscylacje (zbieżność do minimum, nie tylko
w jego okolicę). Równanie normalne: gdy $d$ małe i $X^TX$ dobrze uwarunkowane / odwracalne.

**1.6.** Granice: $\tau\to0^+$ → bardzo lokalne, każde zapytanie „widzi” garstkę najbliższych
punktów (ryzyko przeuczenia / osobliwości $X^TWX$); $\tau\to\infty$ → wszystkie wagi $\to1$ →
zwykłe (globalne) najmniejsze kwadraty. Nieparametryczność: nie streszczamy danych do skończonego
$\theta$ — by przewidywać, trzymamy **cały** zbiór treningowy (rozmiar reprezentacji rośnie z $n$).

**2.1.** Zapis macierzowy jak w 1.3, tylko z $W$ w środku iloczynu; wyprowadzenie identyczne →
$X^TWX\theta=X^TW\vec{y}$. Odwracalność $X^TWX$: np. gdy $w^{(i)}>0$ dla co najmniej $d$ liniowo
niezależnych $x^{(i)}$ (kolumny $X$ liniowo niezależne i dostatecznie wiele dodatnich wag).

**2.2.** $\nabla^2 J=X^TX$; dla dowolnego $v$: $v^TX^TXv=\|Xv\|^2\ge0$ ⇒ $\succeq0$ ⇒ wypukłość.
Ścisła wypukłość ⇔ $Xv=0\Rightarrow v=0$, tj. kolumny $X$ liniowo niezależne (pełny rząd
kolumnowy) ⇔ $X^TX\succ0$ ⇔ jednoznaczne $\hat\theta$. Wypukłość ⇒ każdy punkt stacjonarny jest
minimum globalnym ⇒ GD (przy małym $\alpha$) zbiega globalnie.

**2.3.** $\ell(\theta)=\text{const}-\sum_i\tfrac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma_i^2}$;
maksymalizacja ⇔ minimalizacja $\sum_i\tfrac{1}{\sigma_i^2}(y^{(i)}-\theta^Tx^{(i)})^2$, czyli
ważone NK z $w^{(i)}=1/\sigma_i^2$. Analogia do LWR: lokalne wagi pełnią rolę „zaufania” do
obserwacji — punkty dalekie od $x$ traktujemy jak obarczone większą (efektywną) wariancją dla
lokalnej predykcji, więc ich wkład tłumimy.

**2.4.** Z równania normalnego $X^T(X\hat\theta-\vec{y})=0$ ⇒ $X^T(\vec{y}-\hat{\vec{y}})=0$ (a).
$H=X(X^TX)^{-1}X^T$: $H^2=X(X^TX)^{-1}(X^TX)(X^TX)^{-1}X^T=H$, $H^T=H$ (b) — rzut ortogonalny na
$\mathrm{col}(X)$. Warunek „residuum $\perp\mathrm{col}(X)$” to właśnie równanie normalne; rzut
ortogonalny minimalizuje odległość $\|\vec{y}-X\theta\|$ po $\theta$, stąd „najlepsze” przybliżenie (c).

</details>
