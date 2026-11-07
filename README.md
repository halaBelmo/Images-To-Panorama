# Creation de panorama avec SIFT

Ce projet est une application Streamlit simple qui cree un panorama a partir de deux images qui se chevauchent. Elle utilise OpenCV, SIFT, NumPy et Streamlit.

## Role de SIFT

SIFT detecte des points cles stables dans une image, par exemple des coins, textures ou petits details reconnaissables. Pour chaque point cle, SIFT calcule aussi un descripteur numerique. Ces descripteurs permettent de retrouver les memes zones entre deux images prises depuis des points de vue proches.

## Etapes de l'algorithme

1. Les deux images sont chargees avec OpenCV.
2. Elles sont converties en niveaux de gris.
3. SIFT detecte les points cles et calcule les descripteurs.
4. `BFMatcher` compare les descripteurs avec la distance euclidienne.
5. Le test de ratio de Lowe garde les correspondances les plus fiables.
6. Une validation symetrique supprime beaucoup de fausses correspondances.
7. `findHomography` avec RANSAC/USAC calcule la transformation entre les deux images.
8. Le code teste aussi la direction inverse de l homographie et garde la plus fiable.
9. `warpPerspective` projette la deuxieme image dans le repere de la premiere.
10. Les deux images sont placees dans un canevas commun.
11. Les zones de chevauchement sont fusionnees avec un melange progressif pour rendre le raccord moins visible.

## Installation sur Fedora

Depuis le dossier qui contient `panorama_sift`, lancez :

```bash
cd panorama_sift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Lancement

```bash
cd panorama_sift
source .venv/bin/activate
streamlit run main.py
```

## Utilisation de l'interface

- **Charger image 1** : charge la premiere image depuis la barre laterale.
- **Charger image 2** : charge la deuxieme image depuis la barre laterale.
- **Detecter les points cles** : lance SIFT et affiche les points detectes sur les deux images.
- **Afficher les correspondances** : affiche les correspondances conservees apres le test de Lowe.
- **Generer le panorama** : calcule l'homographie, aligne les images et affiche le panorama final.
- **Telecharger le panorama** : sauvegarde le panorama final en PNG.

Le journal affiche les etapes executees, le nombre de points cles, les correspondances apres filtrage, les inliers RANSAC, le ratio d inliers, l erreur moyenne de reprojection et les erreurs eventuelles.

## Conseils pour obtenir un bon panorama

- Les deux images doivent avoir une zone de chevauchement suffisante.
- La scene doit contenir des details visibles : murs uniformes, ciel ou surfaces lisses donnent peu de points cles.
- Les photos doivent etre prises depuis des positions proches, avec une rotation raisonnable.
- Les objets en mouvement peuvent perturber les correspondances.
- Les changements importants de luminosite peuvent reduire la qualite du resultat.

## Limites

Cette version reste pedagogique, mais elle inclut maintenant un matching plus strict, une validation de qualite et une fusion progressive. Un raccord peut encore rester visible si les photos ont ete prises en se deplacant, si des objets proches dominent la scene, ou si le chevauchement est trop faible. Pour aller encore plus loin, on pourrait ajouter une correction d exposition, une fusion multi-bande ou une gestion avancee de plusieurs images.
