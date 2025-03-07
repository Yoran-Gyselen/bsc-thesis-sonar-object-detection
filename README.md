# Categoriseren van Sonardata met Semi- of Self-supervised Learning

Dit is de repository voor mijn bachelorproef over het categoriseren van sonardata met behulp van semi- of self-supervised learning. Het doel van dit onderzoek is om efficiënte classificatie van sonardata te realiseren door geavanceerde leertechnieken toe te passen.

## Samenvatting

Sinds de opkomst en popularisatie van AI-modellen is data steeds een cruciale re-
source geweest. Voor simpele modellen is de benodigde data vaak ook simpel van
vorm en is er (relatief) weinig van nodig om een performant en goed werkend mo-
del te creëren. Echter stijgen de data-requirements voor grotere en complexere
modellen exponentieel. De benodigde data om een objectdetectiemodel voor so-
nardata te trainen zorgt voor moeilijkheden: dit soort datasets zijn online niet off-
the-shelf beschikbaar en zijn dus zeer tijdrovend en kostbaar om te maken. De
hoofdvraag van dit onderzoek is daarom: Op welke manieren kan het gebruik van
semi- of self-supervised learning het labelproces versnellen zonder een significant
verlies in nauwkeurigheid? Door verschillende technieken toe te passen, zal een
pretraining-strategie ontwikkeld worden die gebruikmaakt van ongelabelde data
om representaties aan te leren. Vervolgens zal onderzocht worden hoeveel gela-
belde data nodig is om een goed presterend detectiemodel te trainen. Het doel is
een methodologie te ontwikkelen die de afhankelijkheid van handmatig gelabelde
data minimaliseert, terwijl de prestaties van het detectiemodel behouden blijven.
De resultaten kunnen bijdragen aan efficiëntere workflows voor data-analyse in so-
narbeeldvorming en andere domeinspecifieke contexten.

## Inhoud van de repository

Deze repository bevat:

- [Het onderzoeksvoorstel](./voorstel/)

- [De bachelorproef zelf](./bachproef/)

- [Een poster](./poster/)

## Installatie en gebruik

Volg hiervoor de [LaTeX@HOGENT gebruikersgids](https://hogenttin.github.io/latex-hogent-gids/). Deze volledige bachelorproef is geschreven op een Ubuntu 24.04.2 LTS systeem, daarom heb ik de aangeraden configuratie voor Linux-systemen gevolgd. Hieronder volgt een korte samenvatting.

### Vereisten

##### Installeer TeX Live en bijhorende software

```bash
sudo apt install biber texlive-extra-utils texlive-fonts-recommended texlive-lang-european texlive-latex-base texlive-latex-extra texlive-latex-recommended texlive-pictures texlive-xetex python3-pygments latexmk
```

##### Installeer TeXstudio om de LaTeX-bestanden te kunnen bewerken

```bash
sudo apt install texstudio
```

##### Configureer TeXstudio

- **Build**:
  - Default compiler: **XeLaTeX** (UTF-8 compatibel, mogelijkheid om TTF-lettertypes te gebruiken, enz.) in plaats van PDFLaTeX (enkel ASCII, PostScript lettertypes, enz.)
  - Default Bibliography Tool: **Biber** (UTF-8 compatibel, ondersteuning voor APA-referenties, ...) in plaats van `bibtex` (enkel ASCII, geen APA-referenties, ...)
- **Commands**:
  - xelatex: `xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %` (voeg de optie `-shell-escape` toe)
- **Editor**:
  - Indentation mode: *Indent and Unindent Automatically*
  - Replace Indentation Tab by Spaces: *Aanvinken*
  - Replace Tab in Text by spaces: *Aanvinken*
  - Replace Double Quotes: *English Quotes: ‘‘’’*
- **Language Checking**:
  - Default language: selecteer "nl_NL-Dutch", "nl-BE" of een gelijkaardige optie voor Nederlands zodat je gebruik kan maken van spellingcontrole. Is deze optie toch niet beschikbaar? Je kan een woordenboekbestand voor OpenOffice of LibreOffice downloaden en installeren. Het dialoogvenster heeft links die je meteen naar de juiste pagina leiden.

##### Installeer JabRef

Download en installeer de `.deb`-file van [jabref.org](https://jabref.org/). De versie in de Debian/Ubuntu-repositories is hopeloos verouderd.

##### Configureer JabRef

- General:
  - Default library mode: **biblatex**
- Entry Preview
  - Zorg er voor dat rechts onder "Selected" ook "American Psychological Association 7th edition" staat.
  - Andere geselecteerde stijlen kan je verwijderen (die hebben wij toch niet nodig), op "Custiomized preview Style" na.
- Linked Files
  - **File directory**, Main file directory: hier kan je het pad naar de *Main file directory* opgeven waar je alle PDF-versies van de artikels en andere bronnen bewaart.
    - Als je zorgt dat alle bestanden in deze directory dezelfde naam hebben als de Bibtex-key van de bron in je bibliografische databank (typisch naam van de eerste auteur + jaartal, bv. Knuth1998.pdf), dan kan JabRef dit automatisch terugvinden en kan je het openen vanuit JabRef zelf.

### Compileren van het document

- Om een LaTeX-document (extensie .tex) te compileren, druk F5 (of kies in het menu voor Tools > Build & View)
  - De eerste keer dat je dit doet kan het compilatieproces lang duren. Er zal je op Windows wellicht gevraagd worden of MikTeX wijzigingen mag aanbrengen aan je systeem. Laat dit toe, want wellicht moeten er nog extra packages geïnstalleerd worden. Dit is eenmalig (per document), dus de volgende keer dat je compileert moet dit een stuk sneller gaan.
  - MiKTeX op Windows zal een pop-up tonen om je toestemming te vragen, bevestig dit. De eerste keer compileren kan enkele minuten duren zonder dat je feedback krijgt over wat er gebeurt. Even geduld, dus.
  - Op Linux werkt dit misschien niet en moet je opzoeken welke extra packages `texlive` je nog moet installeren voor de gewenste functionaliteit.
- Als je een document met een bibliografie wilt compileren, maar die is niet zichtbaar (en verwijzingen zijn in het vet gedrukt, bv. (**Knuth1977**)), dan moet je de bibliografie apart compileren met F8 (of Tools > Bibliography) en daarna opnieuw F5. Nu zou de bibliografie wel toegevoegd moeten zijn!

## Structuur van het project

```
📂 bap-2425-yorangyselen
├── 📂 bachproef         # Alle LaTeX-bestanden & bibliografie van de bachelorproef zelf
├── 📂 fonts             # Vereiste lettertypes voor de template
├── 📂 graphics          # Afbeeldingen en diagrammen
├── 📂 poster            # LaTeX-bestanden voor de poster
├── 📂 voorstel          # LaTeX-bestanden voor het voorstel
├── 📄 README.md         # Dit bestand
├── 📄 .gitattributes    # Specifieke instellingen voor bestanden
└── 📄 .gitignore        # Bestanden die uitgesloten moeten worden uit versiebeheer
```

## Bijdragen

- De vormgeving van het bachelorproefsjabloon is gebaseerd op het werk van [Pieter van der Kloet](https://github.com/pvdk/hogent-latex-thesis). Dat sjabloon wordt nu [hier verder onderhouden](https://github.com/HoGentTIN/latex-hogent-report) door Bert Van Vreckem
- Het sjabloon voor het bachelorproefvoorstel [wordt hier bijgehouden](https://github.com/HoGentTIN/latex-hogent-article).

Volgende personen hebben bijgedragen aan deze sjablonen:

- Bert Van Vreckem
- Chantal Teerlinck
- Jan Willem
- Jens Buysse
- Jeroen Maelbrancke
- Jonas Verhofsté
- Matts Devriendt
- Niels Corneille
- Patrick Van Brussel
- Simon Rondelez

## Contact

Voor vragen of bijdragen, neem contact op via [yoran.gyselen@student.hogent.be](mailto:yoran.gyselen@student.hogent.be).

---

© Yoran Gyselen - 2025
