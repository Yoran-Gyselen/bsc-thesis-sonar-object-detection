# Proof of concept

Deze folder bevat de uitgewerkte PoC die bij de bachelorproef hoort. Specifieke uitleg over de PoC is terug te vinden in de paper zelf.

```
📂 poc
├── 📂 data                         # Klassen die de verschillende types datasets implementeren
├── 📂 models                       # Alle getrainde modellen (gewichten niet inbegrepen wegens grootte, enkel logs)
├── 📂 results                      # Voorspellingen van verschillende modellen op verschillende afbeeldingen
├── 📂 utils                        # Handige helper-functies
├── 📄 byol.py                      # Trainingsscript van de BYOL-backbone
├── 📄 faster_rcnn_byol_optim.py    # Geoptimaliseerd trainingsscript van Faster R-CNN met BYOL-backbone
├── 📄 faster_rcnn_byol.py          # Trainingsscript van Faster R-CNN met BYOL-backbone
├── 📄 faster_rcnn_optim.py         # Geoptimaliseerd trainingsscript van Faster R-CNN
├── 📄 faster_rcnn_paper.py         # Trainingsscript van Faster R-CNN zoals in de paper
├── 📄 fixmatch.py                  # Trainingsscript van Faster FixMatch
├── 📄 README.md                    # Dit bestand
├── 📄 requirements.txt             # Externe packages nodig om deze code te runnen
└── 📄 visualization.ipynb          # Visualisatienotebook voor voorspellingen van modellen
```