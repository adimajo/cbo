# Projet `CBO`

Le package `CBO` est un site web Django (Python) pour l'organisation de petit-déjeuners d'équipe.
Chaque personne a des points ; un ou des organisateur(s) sont tirés aléatoirement (avec une probabilité inversement
proportionnelle à leur nombre de points) ; organiser un petit-déjeuner rapporte des points (+ bonus si fait maison)
et chaque participation enlève des points. Une notification par email est envoyée aux participants et aux organisateurs.

Une chaîne de CI/CD pour Gitlab et Github complète le projet.

## Configuration

Les variables d'environnement suivantes sont nécessaires au démarrage du container / pod :
- SECRET_KEY : clé secrète Django ;
- USERNAME : username du superuser à créer s'il n'existe pas déjà ;
- PASSWORD : password du superuser à créer s'il n'existe pas déjà ;
- TEST_DB : mettre '1' pour utiliser une base sqlite3 locale ;
- TEST_STATIC : use STATICROOT setting? (default: False);
- POSTGRES_URL : URL de la base postgresql ;
- POSTGRES_PORT : port de la base postgresql ;
- POSTGRES_MASTER_DB : nom de la database postgresql ;
- POSTGRES_MASTER_USER : username du compte postgresql ;
- PGPASSWORD : password du compte postgresql ;
- POSTGRES_SCHEMA : schéma spécifique à l'application à créer et ou stocker les tables.

## Développement

### Environnement Python

Le projet utilise **python 3.8** dans un container Docker.

Le projet utilise également **pipenv**.
[Une ressource intéressante](https://moodle.insa-rouen.fr/pluginfile.php/75430/mod_resource/content/4/Python-PipPyenv.pdf).

L'ensemble des dépendances python sont listées dans le fichier `Pipfile`.
Celles-ci peuvent être installées à l'aide de pipenv avec `pipenv install [-d]`.

Pour télécharger l'ensemble des dépendances du projet afin de les porter ensuite 
sur une machine qui disposerait d'un accès limité à internet, il faut utiliser la commande
 `pipenv lock -r > requirements.txt` qui va transformer le `Pipfile` en un `requirements.txt`.

### Utilisation pour téléchargement des dépendances

A partir du fichier `requirements.txt`, il devient facile de télécharger les packages sous la forme 
de `wheels` pour les installer ensuite sur un environnement dépourvu de connexion internet.

Il faut utiliser la commande `pip download -d dossier -r path_to/requirements.txt` où `dossier` représente
le dossier dans lequel on veut stocker les `wheels` et `path_to` désigne le chemin vers le fichier `requirements.txt`
d'intérêt.

### Installation offline ultérieure

L'installation offline à partir des `wheels` préalablement téléchargées se fait avec la commande 
`pip install --no-index --find-links dossier` où `dossier` le dossier dans lequel on vient de
stocker les `wheels`.
