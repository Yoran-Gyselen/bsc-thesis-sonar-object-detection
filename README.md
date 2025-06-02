# Objectdetectie in sonardata met behulp van semi- en self-supervised learning

Dit is de repository voor mijn bachelorproef over objectdetectie in sonardata met behulp van semi- of self-supervised learning. Het doel van dit onderzoek is om een verkennend onderzoek naar het toepassen van moderne leertechnieken op onderwaterbeeldvorming uit te voeren.

## Samenvatting

Sinds de opkomst van krachtige AI- en deep learning-modellen is data uitgegroeid tot een essentiële en vaak beperkende factor in het ontwikkelingsproces. Waar eenvoudige modellen vaak kunnen volstaan met beperkte en eenvoudige datasets, vereisen complexere modellen &ndash; zoals die voor objectdetectie &ndash; steeds grotere en rijkere hoeveelheden gelabelde data. Dit vormt een belangrijk probleem in domeinen zoals sonarbeeldvorming, waar dergelijke datasets niet beschikbaar zijn als kant-en-klare bronnen en handmatige annotatie buitengewoon tijdsintensief en kostbaar is. Dit onderzoek richt zich daarom op de centrale vraag: hoe kunnen semi-supervised en self-supervised leermethoden het labelproces bij objectdetectie in sonardata versnellen, zonder significant verlies aan nauwkeurigheid?

Om deze vraag te beantwoorden is een experimenteel kader opgezet waarin drie benaderingen zijn onderzocht: een volledig supervised baseline gebaseerd op Faster R-CNN, een semi-supervised model met FixMatch, en een self-supervised strategie waarbij een BYOL-model wordt gepretraind en vervolgens gebruikt als backbone binnen een Faster R-CNN-architectuur. De experimenten zijn uitgevoerd op een publieke sonardataset bestaande uit 7600 gelabelde sonarbeelden. Voor de supervised baseline is het model getraind op verschillende hoeveelheden gelabelde data: 1%, 5%, 10%, 50% en 100%. De bijbehorende mAP-scores tonen een sterke daling in nauwkeurigheid naarmate de hoeveelheid gelabelde data afneemt, met resultaten variërend van 0.7717 (100%) tot slechts 0.2799 bij gebruik van 1% van de data.

In het semi-supervised scenario is FixMatch toegepast met 5% en 10% gelabelde data, terwijl de resterende data werd gebruikt als ongelabelde input. Deze aanpak resulteerde in mAP-scores van respectievelijk 0.6649 en 0.6828, wat duidelijk betere prestaties zijn dan het supervised model op dezelfde labelniveaus. Voor het self-supervised model werd BYOL gepretraind op de volledige dataset zonder labels. De representaties die dit opleverde zijn vervolgens geïntegreerd in Faster R-CNN, waarbij opnieuw 5% en 10% van de data gelabeld werd gebruikt voor training. Deze benadering leverde de hoogste nauwkeurigheid binnen de lage-labelscenario's, met mAP-scores van respectievelijk 0.6452 en 0.7230.



De resultaten van dit onderzoek tonen aan dat zowel semi-supervised als self-supervised technieken effectief zijn in het verminderen van de afhankelijkheid van handmatig gelabelde data, terwijl de modelprestaties grotendeels behouden blijven. Met name self-supervised pretraining via BYOL blijkt zeer waardevol in situaties met beperkte gelabelde data. Deze bevindingen bieden praktische aanknopingspunten voor het ontwikkelen van efficiëntere workflows in sonarbeeldanalyse, en zijn relevant voor bredere toepassingen in domeinen waar gelabelde data schaars of moeilijk te verkrijgen is. Hoewel de resultaten veelbelovend zijn, is vervolgonderzoek nodig om de generaliseerbaarheid naar andere types sonardata of real-time toepassingen te evalueren.

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
├── 📂 poc               # De implementatie van de PoC (verschillende modellen, ...)
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
