# Reconnaissance vocale en IPA étroite (phonétique) du mandarin

[![ModelScope](https://img.shields.io/badge/ModelScope-魔搭-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-brightgreen"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

Basé sur l'encodeur [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice), au-dessus duquel est montée une **tête de décodage CTC IPA étroite (narrow/phonetic)** ; par un fine-tuning complet (full fine-tuning), il réalise la transcription vocale du mandarin au niveau « syllabe + ton ».

Le modèle produit une séquence de syllabes IPA étroites séparées par des espaces, chaque syllabe portant son propre symbole de valeur tonale (ex. `ɡ̊wa̠n̚˥`, `x̞wa̠ɪ̯˧˥`) — donnant à la fois l'initiale/finale et le ton, adapté à l'analyse phonétique, l'évaluation de la prononciation mandarine, l'enseignement des tons, etc.

> Licence : **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).
> Les poids du modèle et les données d'entraînement ne sont pas fournis avec le dépôt de code ; ils s'obtiennent séparément via ModelScope et un pipeline local de construction des données. Voir ci-dessous.

---

## Caractéristiques

- **Sortie IPA étroite** : adopte le schéma étroit UntPhesoca de [nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter), étiquetage au niveau phonème tout en conservant les valeurs tonales.
- **Fine-tuning complet** : dégel de l'encodeur Conformer de SenseVoiceSmall et entraînement conjoint avec la nouvelle tête IPA ; le frontend de caractéristiques est gelé, `dither=0` garantit la reproductibilité.
- **Métrique de ton indépendante de l'étiquette** : la précision tonale est calculée via une table de correspondance `ipa2tone` (syllabe IPA → classe de ton 1–5), indépendamment de la notation phonémique spécifique, facilitant une comparaison équitable.
- **Validation à grande échelle** : mesurée sur un ensemble de validation fusionné de 5918 échantillons (voir [Métriques d'évaluation](README.md#效果指标大规模验证)).
- **Démo Gradio** : prend en charge le téléversement/enregistrement audio pour la reconnaissance, et compare automatiquement les différences avec un texte de référence (alignement tonal par syllabe avec surbrillance).

---

## Structure du répertoire

```
mandarin-ipa-asr/
├── LICENSE                                  # Texte intégral de CC BY-NC-SA 4.0
├── README.md
├── requirements.txt                         # Versions des dépendances mesurées
├── .gitignore
├── app.py                                   # Démo Gradio (reconnaissance + comparaison auto des différences)
├── configs/
│   └── example_train_config.json            # Exemple de configuration d'entraînement
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # Vocabulaire de sortie IPA (1427 classes, publié)
│   └── vocab_mandarin_ipa_tone_combined.json# Syllabe IPA -> classe de ton 1-5 (publié)
├── src/
│   ├── model.py        # SenseVoiceIpa : encodeur + tête CTC IPA
│   ├── utils.py        # Décodage CTC / métriques / alignement / beam search / chargement audio
│   ├── dataset.py      # Jeu de données audio-texte (wav/flac/ogg/mp3)
│   ├── train.py        # Entraînement par fine-tuning complet
│   ├── infer.py        # Inférence unique + évaluation par lots (validation à grande échelle)
│   └── prepare_data.py # Construction du jeu d'entraînement IPA à partir du texte pinyin (prétraitement, sans audio)
└── results/
    └── metrics.md       # Métriques de validation mesurées et courbes d'entraînement
```

---

## Modèle de base

| Élément | Description |
| --- | --- |
| Nom | SenseVoiceSmall (FunAudioLLM / Académie DAMO, Alibaba) |
| Source | GitHub : [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) ; ModelScope : `iic/SenseVoiceSmall` |
| Structure | `WavFrontend` (caractéristiques fbank + f0 optionnel) + encodeur Conformer (512 dim) + tête CTC de caractères chinois d'origine |
| Modification de ce projet | Frontend gelé ; une nouvelle **tête CTC IPA** est montée sur son encodeur Conformer (voir ci-dessous) ; l'ancienne tête de caractères chinois n'est plus utilisée |

> Les poids du modèle de base sont automatiquement téléchargés et mis en cache par ModelScope lors de la première exécution ; aucune préparation manuelle requise.

---

## Méthode d'entraînement

1. **Initialisation** : chargement du `WavFrontend` et de l'`encoder` pré-entraînés de SenseVoiceSmall (la tête CTC de caractères chinois d'origine est abandonnée), puis création d'une nouvelle tête CTC IPA `ctc_head = Linear(512,512) → ReLU → Dropout(0.1) → Linear(512, vocab)`.
2. **Stratégie de gel** : `WavFrontend` toujours gelé, `dither` fixé à 0 (désactive l'injection de bruit aléatoire, garantit des caractéristiques déterministes) ; l'encodeur et la tête IPA sont **dégelés et fine-tunés complètement** (l'entraînement de la seule tête est aussi pris en charge).
3. **Perte et optimisation** : `CTCLoss(blank=0)` ; AdamW (`lr=1e-4`, `weight_decay=1e-4`) ; `CosineAnnealingLR` (`T_max = steps × epochs`) ; écrêtement du gradient 1.0.
4. **Précision** : précision mixte bf16 (`autocast` appliqué uniquement à la passe avant de l'encodeur ; caractéristiques et perte restent en fp32).
5. **Sauvegarde des poids** : `best.pt` (TER de validation le plus bas), `best_tone.pt` (précision tonale la plus élevée).
6. **Compatibilité des anciens poids** : lors de l'inférence / reprise d'entraînement, le nom de tête résiduel des anciens poids est automatiquement remappé en `ctc_head.*` pour éviter une perte silencieuse de la tête.

---

## Données d'entraînement

| Élément | Quantité |
| --- | --- |
| Ensemble d'entraînement | **11 691** échantillons (lecture mandarine ; dérivés du sous-ensemble mandarin de zhvoice ~8 993 échantillons réutilisés/étendus à 1,3×) |
| Ensemble de validation | **918** échantillons (lecture mandarine, disjoint de l'ensemble d'entraînement) |
| Étiquettes | Chaque échantillon est une séquence de syllabes IPA étroites séparées par des espaces (convertie du pinyin via nk2028/putonghua-ipa-converter) |
| Vocabulaire IPA | 1 427 classes (incluant `<blank>`/`<unk>`) |

> Le corpus zhvoice brut représente environ **900 heures, 3 200+ locuteurs, ~1 129 800 entrées de texte** ;
> ce projet n'utilise que son sous-ensemble mandarin de haute qualité de lecture pour construire les ensembles d'entraînement/validation, convertis en IPA étroite.

---

## Paramètres du modèle

| Paramètre | Valeur |
| --- | --- |
| Nombre total de paramètres | **234 993 874** |
| Paramètres entraînables (fine-tuning complet) | **222 131 699** (encodeur + tête IPA ; frontend et ancienne tête de caractères chinois gelés) |
| Inférence seule (tête entraînable, encodeur gelé) | 994 707 |
| Dimension de l'encodeur | 512 |
| Vocabulaire de sortie IPA | 1 427 |
| Entrées de correspondance tonale (ipa2tone) | 1 425 (hors `<blank>`/`<unk>`) |

---

## Sources des ensembles de données et liens

| Données / Outil | Usage | Licence | Lien |
| --- | --- | --- | --- |
| **zhvoice** | Corpus d'entraînement (sous-ensemble lecture mandarine) | Voir dépôt | https://github.com/fighting41love/zhvoice |
| **putonghua-ipa-converter** | Conversion pinyin → IPA étroite (scheme 2, UntPhesoca) | CC0 | https://github.com/nk2028/putonghua-ipa-converter |
| **SenseVoiceSmall** | Modèle de base pré-entraîné | Voir dépôt | https://github.com/FunAudioLLM/SenseVoice (ModelScope `iic/SenseVoiceSmall`) |

---

## Métriques d'évaluation (validation à grande échelle)

Le décodage est totalement cohérent avec l'entraînement (entrée forme d'onde brute, dither=0, passe avant encodeur bf16). Les définitions des métriques sont dans [`results/metrics.md`](results/metrics.md).

### 918 échantillons du corpus de validation mandarin de premier niveau (Yijia)

| Métrique | Valeur |
| --- | --- |
| TER (taux d'erreur de syllabe, plus bas est mieux) | **0,0955** |
| ACC (correspondance exacte de la phrase entière) | 0,7059 |
| TACC (précision des tokens) | 0,9145 |
| Précision tonale (indépendante de l'étiquette) | **0,9212** |

### 5918 échantillons de l'ensemble de validation fusionné (validation à grande échelle)

= 5000 mp3 zhvoice en situation réelle + 918 lectures mandarines. zhvoice est en situation réelle, plus diversifié en domaine et plus bruyant, donc son ACC phrase entière inférieure au corpus de validation mandarin Yijia est attendue ; la reconnaissance des tons reste à un niveau élevé.

| Métrique | Valeur |
| --- | --- |
| TER (taux d'erreur de syllabe, plus bas est mieux) | **0,1089** |
| ACC (correspondance exacte de la phrase entière) | 0,4439 |
| TACC (précision des tokens) | 0,8986 |
| Précision tonale (indépendante de l'étiquette) | **0,8915** |

> Remarque : la précision tonale est comparable à des modèles similaires sous un schéma pinyin (même cadre, tête pinyin ton ~0,92, cette tête IPA 0,89–0,92), ce qui démontre qu'avec l'annotation conjointe « phonème + ton », l'information tonale n'est pas perdue.

---

## Installation et environnement (tous mesurés)

| Dépendance | Version | Notes |
| --- | --- | --- |
| Python | **3.9.13** | 3.9 recommandé (3.8–3.11 devraient fonctionner ; 3.9.13 mesuré) |
| CUDA | **12.8** | Entraînement/inférence nécessite un GPU NVIDIA ; inférence CPU seule possible mais lente |
| torch | 2.7.0+cu128 | correspond à CUDA 12.8 |
| torchaudio | 2.7.0+cu128 | |
| funasr | 1.4.16 | charge SenseVoiceSmall |
| modelscope | 1.32.0 | télécharge automatiquement le modèle de base |
| transformers | 4.43.0 | |
| numpy | 1.23.4 | |
| editdistance | 0.6.2 | calcul des métriques |
| tqdm | 4.64.1 | barre de progression |
| soundfile | 0.12.1 | lecture/écriture audio |
| gradio | 4.24.0 | interface de démo |
| librosa | 0.9.2 | (optionnel) prétraitement des données |
| ffmpeg | 2025-08-23 | décodage mp3 (doit être dans le PATH) |

Installation :

```bash
pip install -r requirements.txt
# ffmpeg doit être installé séparément et ajouté au PATH (Windows : https://www.gyan.dev/ffmpeg/ ou scoop/apt)
```

**Exigences matérielles** : entraînement recommandé ≥ 16 Go de VRAM (fine-tuning complet de 234M paramètres + activations en bf16) ; inférence nécessite seulement quelques Go, un seul échantillon audio peut tourner sur CPU (lent).

---

## Démarrage rapide

> Tous les scripts résolvent `vocab/`, `data/`, `weights/`, `checkpoints/` et autres chemins relatifs par rapport à la « racine du projet » (c.-à-d. la racine du dépôt), donc **pas besoin de `cd` dans la racine du projet pour lancer depuis n'importe quel répertoire de travail** ; un chemin absolu est utilisé tel quel.
> Le modèle de base `iic/SenseVoiceSmall` est un id de modèle ModelScope et sera téléchargé et mis en cache automatiquement lors de la première exécution.

> ### 【Important】Vous devez charger les poids fine-tunés d'abord, sinon la sortie est un gibberish sans sens
>
> Ce projet **ne place pas les poids du modèle dans le dépôt** (publié séparément sur ModelScope). Si vous démarrez la démo ou l'inférence **sans spécifier les poids via `--ckpt`**,
> le programme utilisera silencieusement un modèle **initialisé aléatoirement** — l'IPA qu'il émet pour l'audio sera sans sens et présentera souvent des syllabes répétées,
> ressemblant à « beaucoup a été reconnu », mais ce n'est pas du tout la prononciation réelle.
> Par exemple, la première moitié de *Jingyesi* (静夜思) n'a que ~10 syllabes, mais sans poids elle peut produire 30+ syllabes répétées absurdes.
>
> Usage correct (les poids doivent d'abord être téléchargés vers `weights/best.pt`, voir section suivante) :
> ```bash
> python app.py --ckpt weights/best.pt
> ```
> Si `--ckpt` n'est pas donné, `app.py` tentera automatiquement de localiser `weights/best.pt` ; si les deux sont absents, un 【avertissement】proéminent est affiché au démarrage.

### 1. Obtenir les poids du modèle

Les poids du modèle **ne sont pas dans ce dépôt** ; téléchargez-les depuis ModelScope (publiés séparément par l'auteur), par ex. :

```bash
# En supposant que le modèle est publié, utilisez modelscope pour télécharger localement
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

Après avoir obtenu `weights/best.pt`, spécifiez-le via `--ckpt`. (Si vous entraînez le vôtre, un `checkpoints/best_tone.pt` supplémentaire sera produit ; voir « Guide d'entraînement » ci-dessous)

### 2. Inférence sur un audio

```bash
# Pas besoin de cd dans la racine du projet : le script résout vocab/data relativement à sa propre position ; lançable depuis n'importe quel répertoire
python src/infer.py --wav path/to/audio.wav --ckpt weights/best.pt
# Sortie : séquence de syllabes IPA étroites séparées par des espaces
```

### 3. Évaluation par lots (métriques de validation)

```bash
python src/infer.py --eval --ckpt weights/best.pt \
    --val_scp data/val.scp --val_text data/val.text --limit 0
# Sortie TER / ACC / TACC / précision tonale
```

### 4. Démo Gradio (téléverser audio + comparaison automatique des différences)

> Chargez absolument les poids via `--ckpt` (voir la note 【Important】 ci-dessus), sinon le résultat de reconnaissance est un gibberish aléatoire sans sens.
> L'onglet « Reconnaître » utilise par défaut la recherche par faisceau (beam search, `--beam`, défaut 12) pour réduire les erreurs d'insertion/répétition ; `--beam 0` revient au décodage glouton (greedy).

```bash
# Lançable depuis n'importe quel répertoire (chemins relatifs résolus par rapport à la racine du projet) ; --ckpt accepte un chemin relatif ou absolu
# Si --ckpt est omis, weights/best.pt est tenté automatiquement
python app.py --ckpt weights/best.pt --port 7860
# Ouvrez http://127.0.0.1:7860 dans un navigateur
```

- **Onglet Reconnaître** : téléversez ou enregistrez un audio → produit l'IPA étroite.
- **Onglet Comparer (différence auto)** : téléversez un audio et (facultatif) saisissez le texte IPA de référence →
  effectue automatiquement un alignement tonal par syllabe et surligne les différences (correct / ton erroné / mal lu / en trop / manquant) ;
  même sans texte de référence, il compare automatiquement les deux résultats de décodage « glouton vs faisceau ».

---

## Guide d'entraînement

1. Préparez les données (voir « Notes d'utilisation des données » ci-dessous) pour obtenir `train.scp` / `train.text` / `val.scp` / `val.text`
   (format : `uid chemin_audio` et `uid syllabes_IPA_séparées_par_espaces`).
2. Exécutez :

```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

Vous pouvez aussi écrire les paramètres dans `configs/example_train_config.json` puis simplement exécuter :

```bash
python src/train.py $(python -c "import json,sys; c=json.load(open('configs/example_train_config.json')); print(' '.join(f'--{k} {v}' for k,v in c.items()))")
```

- Pour un warm-start à partir de poids fine-tunés pinyin : `--warm_start weights/pinyin_ft.pt` (dimension de tête différente est automatiquement ignorée et initialisée aléatoirement).
- Produit `checkpoints/best.pt` (TER le plus bas) et `checkpoints/best_tone.pt` (ton le plus élevé).

---

## Notes d'utilisation des données (données non publiques)

Pour des raisons de licence et de taille, **les données d'entraînement ne sont pas publiées directement**. Vous pouvez reconstruire un ensemble de données équivalent en suivant ces étapes :

1. Téléchargez le corpus **zhvoice** (https://github.com/fighting41love/zhvoice), décompressez pour obtenir l'audio et le texte pinyin.
2. Clonez **putonghua-ipa-converter** (https://github.com/nk2028/putonghua-ipa-converter), utilisez son `data/putonghua.js` (scheme 2 = UntPhesoca étroit) pour convertir le pinyin en IPA étroite.
3. Utilisez `src/prepare_data.py` de ce dépôt pour générer les fichiers nécessaires à l'entraînement :

```bash
# 1) sous-ensemble zhvoice -> texte IPA étroit + scp + vocabulaire
#    --conv_js pointe vers le data/putonghua.js du convertisseur ; --audio_root est le répertoire racine audio
python src/prepare_data.py build-zhvoice \
    --metadata zhvoice/metadata.csv \
    --audio_root zhvoice/wavs \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa --train_n 8993 --val_n 918

# 2) convertir votre propre texte pinyin à tons en IPA (le répertoire doit contenir train/text, val/text)
python src/prepare_data.py build-mandarin \
    --text_dir data/mandarin_pinyin \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa

# 3) fusionner plusieurs sources et étendre le vocabulaire, sortir les fichiers *_combined
python src/prepare_data.py combine \
    --zhvoice_dir data/zhvoice_ipa --mandarin_ipa_dir data/zhvoice_ipa \
    --out data/combined

# 4) étendre l'ensemble d'entraînement par un facteur (ex 1,3×) : base sur le sous-ensemble mandarin, mélanger les données zhvoice
python src/prepare_data.py scale \
    --mandarin_text data/combined/train_text --mandarin_scp data/combined/train_scp \
    --zhvoice_text data/zhvoice_ipa/train_text --zhvoice_scp data/zhvoice_ipa/train_scp \
    --factor 1.3 --out data/ipa130
```

Les `*.scp` (chemins audio) et `*.text` (étiquettes IPA) générés peuvent servir d'entrée à `train.py` / `infer.py`.
**Ne publiez pas l'audio brut ou le texte tiers avec ce dépôt** ; publiez uniquement les scripts et les notes d'utilisation ci-dessus.

---

## Licence

Le code et le vocabulaire sont publiés sous **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ; voir [LICENSE](LICENSE)).
- **Attribution (BY)** : veuillez conserver l'auteur original et l'origine du projet lors de l'utilisation.
- **NonCommercial (NC)** : ne peut pas être utilisé à des fins commerciales.
- **ShareAlike (SA)** : les œuvres dérivées doivent être publiées sous la même licence.

Les poids du modèle et les données d'entraînement sont fournis séparément sous leurs licences sources respectives (ModelScope / dépôts de sources de données) et ne remplacent pas la licence de ce dépôt.

---

## Citation et remerciements

- Modèle de base : FunAudioLLM, *SenseVoice*.
- Conversion IPA étroite : nk2028, *putonghua-ipa-converter* (CC0).
- Corpus d'entraînement : fighting41love, *zhvoice*.
