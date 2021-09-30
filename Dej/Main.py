# -*- coding: utf-8 -*-
# ================================================= Import des packages ================================================
# Operationg system
import os
import io
os.environ["PATH"] = os.environ["PATH"] +r";N:\Projets02\PRESTATIONS DE SERVICE\04 - LCL\2018 Promotion immobilière\04 - Programmes\Dépendance\GraphViz\release\bin"
import base64
import urllib
# Gestion des dates
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

# Gestion de données
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

# Affichage de graphes
import bokeh
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show, save as save_bokeh
from bokeh.layouts import column, row
from bokeh.models import Legend, ColumnDataSource, LinearAxis, Range1d, Band
from bokeh.resources import INLINE
from bokeh.palettes import viridis
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
RESS2 = bokeh.resources.Resources('cdn')

# Pages HTML
import jinja2

# Homemade packages
from DjangoSite.Singleton import Singleton
from Promotion.Print import MyPrinter
from Promotion.wordCloud import CloudHandler
from Promotion.utils import human_format, human_million, analyse_corr, analyse_corr_2, \
    to_float, diff_month, apply_filter, to_html_string

# Modélisation
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

# Visusalisation des arbres de décision
import pydotplus

# Autre
import pprint


# ================================================= Définition des discretizers ========================================
class MyKBinsDiscretizer(object):
    def __init__(self,
                 n_bins=4,
                 encode="ordinal",
                 strategy="quantile",
                 *args,
                 **kwargs):
        self.enc = KBinsDiscretizer(n_bins=n_bins,
                                    encode=encode,
                                    strategy=strategy,
                                    *args, **kwargs)

    def fit(self, X, *args, **kwargs):
        if isinstance(X, pd.Series):
            X = X.values
        X = X[~np.isnan(X)]

        self.mapping_ = {-1: np.nan}
        for idx, elt in enumerate(np.percentile(X, q=list(np.linspace(0, 100, self.enc.n_bins)))):
            self.mapping_[idx] = elt

        X = X.reshape(-1, 1)

        self.enc.fit(X, *args, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, np.ndarray):
            X = pd.Series(X)
        row_filter = ~np.isnan(X)
        X.loc[row_filter] = self.enc.transform(X.loc[row_filter].values.reshape(-1, 1), *args, **kwargs).reshape(-1)
        X = X.fillna(-1)
        return X.values

    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X, *args, **kwargs).transform(X)

    def inverse_transform(self, X):
        # row_filter = X == -1
        # X[~row_filter] = self.enc.inverse_transform(X[~row_filter].reshape(-1, 1)).reshape(-1)
        # X[row_filter] = np.nan
        return [self.mapping_[elt] for elt in X]

class DayTransformer(object):
    def __init__(self, *args, **kwargs):
        self.today = date.today()

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        df = pd.Series(X)
        row_filter_nan_dates = df.notnull()
        df.loc[row_filter_nan_dates] = df.loc[row_filter_nan_dates].apply(lambda x: (self.today - x).days)
        df.loc[~row_filter_nan_dates] = -1
        return df.values

    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X, *args, **kwargs).transform(X, *args, **kwargs)

class MonthTransformer(object):
    def __init__(self, *args, **kwargs):
        self.today = date.today()

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        df = pd.Series(X)
        row_filter_nan_dates = df.notnull()
        df.loc[row_filter_nan_dates] = df.loc[row_filter_nan_dates].apply(lambda x: diff_month(x, self.today))
        df.loc[~row_filter_nan_dates] = -1
        return df.values

    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X, *args, **kwargs).transform(X, *args, **kwargs)

class DayBinsTransformer(object):
    def __init__(self, *args, **kwargs):
        self.kbin_discretize = MyKBinsDiscretizer(*args, **kwargs)
        self.today = date.today()

    def fit(self, X, *args, **kwargs):
        if isinstance(X, pd.Series):
            X = X.values
        df = pd.Series(X)
        row_filter_nan_dates = df.notnull()
        values = df.loc[row_filter_nan_dates].apply(lambda x: (self.today - x).days)
        return self.kbin_discretize.fit(values, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.Series):
            X = X.values
        df = pd.Series(X)
        row_filter_nan_dates = df.notnull()
        df.loc[row_filter_nan_dates] = df.loc[row_filter_nan_dates].apply(lambda x: (self.today - x).days)
        return self.kbin_discretize.transform(df.values, *args, **kwargs)

    def inverse_transform(self, X):
        return self.kbin_discretize.inverse_transform(X)

    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X, *args, **kwargs).tranform(X, *args, **kwargs)

# ================================================= Définition des etapes de modélisation ==============================
def several_tries_nb_cluster(X, values_to_try, repetitions):
    """
    Given a dataset, tries several values for a Kmean classification with "repetitions" repetitions for each values.
    Outputs Key Points Indicators (KPIs) for each tested value
    """
    # Seed
    np.random.seed(5)  # Graine aléatoire --> Assurer la reproductibilité des résultats

    dict_result = {k: [] for k in ["nb_cluster", "min_size", "max_size", "silhouette"]}
    # Prepare the loop
    inc_ = 0
    # Main loop
    for n_clusters in values_to_try:
        for _ in range(repetitions):
            inc_ += 1
            print("Tentative n°{} - nombre de clusters : {}".format(inc_, n_clusters))

            # Fit the model
            estKmeans = KMeans(n_clusters=n_clusters)
            estKmeans.fit(X)

            # Cimpute the size of each cluster
            sizeClusters = np.array([sum(estKmeans.labels_ == k) for k in range(n_clusters)])

            # Stats
            dict_result["nb_cluster"].append(n_clusters)
            dict_result["min_size"].append(100.0 * min(sizeClusters) / len(X))
            dict_result["max_size"].append(100.0 * max(sizeClusters) / len(X))
            dict_result["silhouette"].append(
                silhouette_score(X, estKmeans.labels_, metric='euclidean', sample_size=1000))

    # Reshape the data
    df_result = pd.DataFrame(dict_result)
    return df_result

def visualise_kopt(df_result):
    """
    Plot a graph using the results of the previously defined function

    - For a given k:
        - Distribution of the maximum size
        - Median of the maximum sizes
        - Distribution of the minimum size
        - Median of the minimum sizes
        - The disribution of the silhouette scores
        - The median silhouette scores
    """
    df_result.sort_values(by="nb_cluster", inplace=True, ascending=True)
    p = figure(width=500, tools="pan, wheel_zoom, box_zoom, save, reset")

    # Setting the second y axis range name and range
    p.extra_y_ranges = {"silhouette": Range1d(start=0, end=1)}

    # Adding the second axis to the plot.
    p.add_layout(LinearAxis(y_range_name="silhouette", axis_label='silhouette'), 'right')

    df_agged = df_result.groupby(["nb_cluster"]).agg({"min_size": {"Plus petite des tailles minimales": np.min,
                                                                  "Médiane des tailles minimales": np.median,
                                                                  "Plus grande des tailles minimales": np.max},
                                                     "max_size": {"Plus petite des tailles maximales": np.min,
                                                                  "Médiane des tailles maximales": np.median,
                                                                  "Plus grande des tailles maximales": np.max},
                                                      "silhouette": {"Plus petite des silhouettes": np.min,
                                                                    "Médiane des silhouettes": np.median,
                                                                    "Plus grande des silhouettes": np.max}})
    df_agged.columns = df_agged.columns.droplevel(0)
    df_agged = df_agged.reset_index()
    print(df_agged)

    _vars = [("Médiane des tailles minimales", "red", None),
             ("Médiane des tailles maximales", "black", None),
             ("Médiane des silhouettes", "green", "silhouette"),
             ]
    for _name, _color, _ax_name in _vars:
        _kwargs = {"x": df_agged["nb_cluster"],
                   "y": df_agged[_name],
                   "color": _color}
        if _ax_name is not None:
            _kwargs["y_range_name"] = _ax_name
        p.line(**_kwargs)
        p.circle(legend=_name, **_kwargs)

        source = ColumnDataSource(df_agged)
        if _ax_name is not None:
            _kwargs = {"y_range_name": _ax_name}
        else:
            _kwargs = {}

        band = Band(base="nb_cluster",
                    lower=_name.replace("Médiane", "Plus petite"),
                    upper=_name.replace("Médiane", "Plus grande"),
                    source=source,
                    level='underlay',
                    fill_color=_color,
                    **_kwargs)
        p.add_layout(band)
    return p

def get_desired_n_clusters():
    """
    Ask the user for the desired number of clusters.

    This number must be positiv and less than the number of samples

    The choice can be based on the graphs produces by the above-defined functions.
    """
    while True:
        try:
            nb_cluster = int(input("Nombre de classes?"))

            if 0 < nb_cluster < len(df):
                return nb_cluster
            else:
                print("The number of clusters must be positive and smaller than the number of samples")
        except Exception as e:
            print(e)

def fit_model(X, n_clusters, *args, **kwargs):
    """
    fit a Kmean model
    return the model.
    """
    model = KMeans(n_clusters=n_clusters, *args, **kwargs).fit(X)
    return model

def get_labels(X, model):
    """
    Return the labels associated with each sample used during the calibration of the model.
    If the sample contains a "nan" value, the label is forced to -1
    """
    labels = model.labels_
    labels[np.isnan(X).sum(axis=1) > 0] = -1
    return labels

def construct_decision_tree(X, y, columns, *args, **kwargs):
    """
    x_learn : data used to fit the cluster
    labels : clusters
    columbs : Name of the variables in the order of x_learn
    """
    clf = tree.DecisionTreeClassifier(*args, **kwargs)
    clf = clf.fit(X, y)

    dot_file = io.StringIO()
    tree.export_graphviz(clf,
                         out_file=dot_file,
                         feature_names=columns,
                         filled=True,
                         rounded=True,
                         special_characters=True,
                         proportion=True,
                         rotate=True)
    graph = pydotplus.graph_from_dot_data(dot_file.getvalue())
    #display(Image(graph.create_png()))
    return clf, graph

#=========================================== Param init ===============================================================#
# Config
if False:

    config = {
        "path_data": r"N:\Projets02\PRESTATIONS DE SERVICE\04 - LCL\2018 Promotion immobilière\03 - Données\pipeline_GRO_v0_retouché.xlsx",
        "path_dir_template": r"N:\Projets02\PRESTATIONS DE SERVICE\04 - LCL\2018 Promotion immobilière\04 - Programmes\DjangoSite\Promotion",
        "path_graphviz": r";C:\Users\DAMAYNI\Desktop\GraphViz\release\bin",
        "total_width": 800,
        "data_processing": {"activate_analysis": False,
                            "limit_size_monovariee": 5000,
                            "limit_size_corr": 1000, },
        "modeling": {"min_percent_cluster": 3.0,
                     "nb_trie_size_cluster_stability": 5}
    }
    path_data = config["path_data"]
    os.environ["PATH"] = os.environ["PATH"] + config["path_graphviz"]

    #=============================================== Data Loadng  =========================================================#
    df = pd.read_excel(config["path_data"])
    df.head(5)

    df_axis= pd.read_excel("Exemple_axes_definition.xlsx")
    relevant_vars = list(df_axis["Variable"].values)
    df = df[relevant_vars]
    #traitement2vars = df_axis.groupby("Traitement")["Variable"].agg(list).to_dict()
    axe2vars = df_axis.groupby("Axe")["Variable"].agg(list).to_dict()
    axes = list(axe2vars.keys())

    traitement2func = {"Mois": MonthTransformer,
                       "Jour": DayTransformer,
                       "Jour + Quantile": DayBinsTransformer,
                       "Quantile": MyKBinsDiscretizer,
                       }

    # Params
    values_to_try = [2, 3, 4, 5]
    repetitions = 10

    var2scaler = {}
    traitement2minmax = {}


    for axe, vars in axe2vars.items():

        row_filter = df_axis["Variable"].isin(vars)
        sub_df_axis = df_axis[row_filter].copy()
        traitement2vars = sub_df_axis.groupby("Traitement")["Variable"].agg(list).to_dict()
        try:
            vars2log = sub_df_axis.groupby("Log")["Variable"].agg(list).to_dict()[True]
        except:
            vars2log = []

        # Creation d'un sous dictionnaire pour cet axe
        var2scaler[axe] = {}
        traitement2minmax[axe] = {}
        break



        #========================= Normalisation des variables =====================
        for traitement, vars in traitement2vars.items():
            func = traitement2func[traitement]
            for var in vars:
                break
            break
                print("Transforming {}".format(var))
                scaler = func()
                var2scaler[axe][var] = scaler
                row_filter = ~np.isnan(df[var])
                X = df[row_filter][var].values
                a = scaler.fit_transform(X)
                b = scaler.inverse_transform(a)

                a = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1)
                b = scaler.inverse_transform(a.reshape(-1, 1)).reshape(-1)
        for var in vars2log:
            df[var] = np.sign(df[var]) * np.log(1 + np.abs(df[var]))

        traitement2minmax = {}
        for traitement, vars in traitement2vars.items():
            if traitement not in ["Quantile", "Jour + Quantile"]:
                scaler = MinMaxScaler((-1, 1))
                traitement2minmax[axe][traitement] = scaler
                df[vars] = scaler.fit_transform(df[vars])

        df[relevant_vars].to_excel("test2.xlsx")


        #========================= Kmeans  =====================
        X = df[vars].values

        MyPrinter.info("Plusieurs tentatives pour k", 1)
        df_result = several_tries_nb_cluster(X, values_to_try=values_to_try, repetitions=repetitions)

        MyPrinter.info("Visualisation des tentatives pour k")
        p = visualise_kopt(df_result)
        show(p)

        n_clusters = get_desired_n_clusters()

        MyPrinter.info("Calibration du modèle retenu, k = {}".format(n_clusters), 1)
        model = fit_model(X, n_clusters)

        MyPrinter.info("Récuperation des labels")
        labels = get_labels(X, model)
        df["cluster_{}".format(axe)] = labels

        #========================= Inverse-Normalisation des variables =====================
        for traitement, vars in traitement2vars.items():
            try:
                scaler = traitement2minmax[traitement]
                df[vars] = scaler.inverse_transform(df[vars])
            except:
                pass

        # Inverse LOG
        for var in vars2log:  #TODO: Se limiter aux variables de l'axe
            df[var] = np.sign(df[var]) * (np.exp(np.abs(df[var])) - 1)

        for traitement, vars in traitement2vars.items():
            if traitement in ["Quantile", "Jour + Quantile"]:
                scaler = var2scaler[var]
                df[var] = scaler.inverse_transform(df[var].values)
        df[relevant_vars].to_excel("test3.xlsx")

        #========================= Decision tree =====================
        row_filter = np.isnan(df[vars].values).sum(axis=1) == 0
        sub = df[row_filter]
        final_tree, graph = construct_decision_tree(sub[vars].values, sub["cluster_{}".format(axe)].values, vars, max_depth=5, min_impurity_split=0.01)
        def graph_to_html(graph):
            return "data:image/png;base64," + urllib.parse.quote(base64.b64encode(graph.create_png()))
        graph_str = graph_to_html(graph)

        break


if True:

    df = pd.read_excel("LastDF.xlsx")


    from bokeh.plotting import figure
    from bokeh.models import (CategoricalColorMapper, HoverTool,
                              ColumnDataSource, Panel,
                              FuncTickFormatter, SingleIntervalTicker, LinearAxis)
    from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,
                                      Tabs, CheckboxButtonGroup,
                                      TableColumn, DataTable, Select)
    from bokeh.layouts import column, row, WidgetBox
    from bokeh.palettes import Category20_16


    # Make plot with histogram and return tab
    def histogram_tab(df):

        name_var = "Nom du/des promoteur(s)"

        # Function to make a dataset for histogram based on a list of carriers
        # a minimum delay, maximum delay, and histogram bin width
        def make_dataset(promoteur_list, n_bins=5):

            # Dataframe to hold information
            by_promoteur = pd.DataFrame(columns=['proportion', 'left', 'right',
                                               'f_proportion', 'f_interval',
                                               'name', 'color'])


            # Iterate through all the carriers
            for i, promoteur_name in enumerate(promoteur_list):
                # Subset to the carrier
                subset = df[df[name_var] == promoteur_name]

                # Create a histogram with 5 minute bins
                arr_hist, edges = np.histogram(subset['CA HT'],
                                               bins=n_bins)

                # Divide the counts by the total to get a proportion
                arr_df = pd.DataFrame({'proportion': arr_hist / np.sum(arr_hist), 'left': edges[:-1], 'right': edges[1:]})

                # Format the proportion
                arr_df['f_proportion'] = ['%0.5f' % proportion for proportion in arr_df['proportion']]

                # Format the interval
                arr_df['f_interval'] = ['%d to %d CA HT' % (left, right) for left, right in
                                        zip(arr_df['left'], arr_df['right'])]

                # Assign the carrier for labels
                arr_df['name'] = promoteur_name

                # Color each carrier differently
                arr_df['color'] = Category20_16[i]

                # Add to the overall dataframe
                by_promoteur = by_promoteur.append(arr_df)

            # Overall dataframe
            by_promoteur = by_promoteur.sort_values(['name', 'left'])

            return ColumnDataSource(by_promoteur)

        def style(p):
            # Title
            p.title.align = 'center'
            p.title.text_font_size = '20pt'
            p.title.text_font = 'serif'

            # Axis titles
            p.xaxis.axis_label_text_font_size = '14pt'
            p.xaxis.axis_label_text_font_style = 'bold'
            p.yaxis.axis_label_text_font_size = '14pt'
            p.yaxis.axis_label_text_font_style = 'bold'

            # Tick labels
            p.xaxis.major_label_text_font_size = '12pt'
            p.yaxis.major_label_text_font_size = '12pt'

            return p


        def make_plot(src):
            # Blank plot with correct labels
            p = figure(plot_width=700, plot_height=700,
                       title='Histogram of CA HT by Promoteur',
                       x_axis_label='CA HT', y_axis_label='Proportion')

            # Quad glyphs to create a histogram
            p.quad(source=src, bottom=0, top='proportion', left='left', right='right',
                   color='color', fill_alpha=0.7, hover_fill_color='color', legend='name',
                   hover_fill_alpha=1.0, line_color='black')

            # Hover tool with vline mode
            hover = HoverTool(tooltips=[('Promoteur', '@{}'.format('name')),
                                        ('CA', '@f_interval'),
                                        ('Proportion', '@f_proportion')],
                              mode='vline')

            p.add_tools(hover)

            # Styling
            p = style(p)

            return p

        def update(attr, old, new):
            carriers_to_plot = [promoteur_selection.labels[i] for i in promoteur_selection.active]

            new_src = make_dataset(carriers_to_plot,
                                   n_bins=binwidth_select.value)

            src.data.update(new_src.data)

        # Carriers and colors
        available_promoteur = list(set(df[name_var]))
        available_promoteur.sort()

        promoteur_selection = CheckboxGroup(labels=available_promoteur,
                                          active=[0, 1])
        promoteur_selection.on_change('active', update)

        binwidth_select = Slider(start=10, end=30,
                                 step=1, value=15,
                                 title='Nombre de bins')

        binwidth_select.on_change('value', update)

        # Initial carriers and data source
        initial_promoteurs = [promoteur_selection.labels[i] for i in promoteur_selection.active]

        src = make_dataset(initial_promoteurs,
                           n_bins=binwidth_select.value)
        p = make_plot(src)

        # Put controls in a single element
        controls = WidgetBox(promoteur_selection, binwidth_select)

        # Create a row layout
        layout = row(controls, p)

        # Make a tab with the layout
        tab = Panel(child=layout, title='Histogram')

        return tab



    tab = histogram_tab(df)
    show(tab)



if False:

    context["df_html"] = df.head(10).to_html(classes="table")

    # Filtre sur les colonne entièrement vides
    to_keep = df.columns[df.count().values >= 0.1 * len(df)]
    if len(to_keep.values) != len(df.columns):
        context["colonnes_vides"] = [x for x in df.columns if not x in to_keep]
    df = df[to_keep]
    df.head(5)


    #
        # =============================================== Data discovery =======================================================#
    # Convert to number
    for var in df.select_dtypes(include=['O', 'int64', 'int', 'float', 'float64']).columns:
        print("Try converting to number : {}".format(var))
        try:
            replacement = df[var].apply(to_float)
            df[var] = replacement
            print("\tDone!")
        except Exception as e:
            print("\tFailed to convert to number : {}".format(e))

    # Statistics per column
    df_data_exploration = pd.DataFrame({"Variable": df.columns,
                                        "Type": df.dtypes,
                                        "Fully_filled": df.count().values == len(df),
                                        "Filled %": [round(x, 2) for x in 100 * df.count().values / len(df)],
                                        "Distinct values count": [len(df[var].value_counts()) for var in
                                                                  df.columns.values],
                                        "Majority value": [df[var].value_counts().head(1).index[0] for var in
                                                           df.columns.values],
                                        "Majority value count": [df[var].value_counts().head(1).values[0] for var in
                                                                 df.columns.values],
                                        "Majority value %": [
                                            round(100 * df[var].value_counts().head(1).values[0] / len(df), 2) for var
                                            in df.columns.values],
                                        })
    context["df_data_exploration"] = df_data_exploration.to_html(classes="table")

    if True:
        if len(object_vars) > 0:
            context["object_vars"] = {}
        for var in object_vars:
            try:
                cloud_handler = CloudHandler()
                cloud_handler.generate(" ".join(df[var].replace(" ", "-")))
                context["object_vars"][var] = {"cloud": cloud_handler.to_html_string()}
                sub = df[var].value_counts().reset_index(name="effectif").head(10)
                sub.columns = [var, "effectif"]
                source = ColumnDataSource(sub)
                col = [TableColumn(field=var, title=var), TableColumn(field="effectif", title="Effectif")]
                data_table = DataTable(source=source, columns=col, width=config["total_width"], height=300)
                script, data_table = components(data_table)
                context["bokeh_scripts"].append(script)
                context["object_vars"][var]["table"] = data_table
            except:
                pass

    if len(date_vars) > 0:
        fig_date_vars = figure(  # y_axis_type="log",
            x_axis_type="datetime",
            title="Effectif par trimestre",
            width=config["total_width"],
            height=800,
            tools=["pan", "wheel_zoom", "box_zoom", "save", "reset"])
        fig_date_vars.xaxis.axis_label = 'Date'
        legend_it = []
        stats = None

    vir = viridis(len(date_vars)+1)
    for idx_color, var in enumerate(date_vars):
        df_copy = df[[var]].copy().dropna()
        row_filter = df_copy[var].apply(lambda x: (x.year >= 2015) and (x.year <= 2030))
        df_copy = apply_filter(df=df_copy,
                           row_filter=row_filter,
                           name_filter="Date postérieures à 2015 et antérieures à 2030")

        mois_trimestrialise = lambda x: 3 * ((x-1) // 3 + 1)  # Fin du trimestre
        df_copy[var] = df_copy[var].apply(lambda x: date(x.year, mois_trimestrialise(x.month), 1)) + relativedelta(months=1, days=-1)
        df_copy_agg = df_copy.groupby(var).size().reset_index(name="effectif")
        kwargs = {"line_width": 2,
                  "y": df_copy_agg["effectif"],
                  "x": df_copy_agg[var],
                  "color": vir[idx_color]}

        fig_date_vars.line(line_dash="dashed", **kwargs)
        c = fig_date_vars.circle(**kwargs)
        legend_it.append((var, [c]))

        # =================== Compute statistics ===================#
        today = date.today()
        row_filter_nan_dates = df[var].notnull()
        df.loc[row_filter_nan_dates, var] = df.loc[row_filter_nan_dates, var].apply(lambda x: diff_month(x, today))

        values = df[row_filter_nan_dates][var]
        stat = values.describe(percentiles=list(np.linspace(0, 1, 4))).reset_index()
        stat.columns = ["Quantile", var]
        stat[var] = stat[var].apply(lambda x: round(x, 2))

        if stats is None:
            stats = stat
        else:
            stats = pd.merge(stats, stat, how="left", on="Quantile")

    # Add legend
    legend = Legend(items=legend_it)
    legend.click_policy = "mute"
    fig_date_vars.add_layout(legend, 'below')

    # Add DataTable
    sub_stats = stats.set_index("Quantile").transpose().reset_index()
    name_cols = list(sub_stats.columns)
    name_cols[0] = "Variable (en nb de mois / aujourd'hui)"
    sub_stats.columns = name_cols
    source = ColumnDataSource(sub_stats)
    cols = []
    for col in sub_stats.columns:  # Larger width for the variable column
        if col == name_cols[0]:  # Variable de date --> Plus de place pour bien lire
            current_width = int(0.4*config["total_width"])
        else:
            current_width = int(0.4*config["total_width"]/(len(sub_stats.columns)-1))
        cols.append(TableColumn(field=col, title=col, width=current_width))
    data_table = DataTable(source=source, columns=cols, width=config["total_width"], height=1000)

    # To jinja format
    fig = row(fig_date_vars, data_table)
    script, fig = components(fig)
    context["bokeh_scripts"].append(script)
    context["date_vars"] = fig


    if True:
        if len(float_vars) > 0:
            context["float_vars"] = {}
        for var in float_vars:
            values = df[var].dropna().astype(float)
            stat = values.describe(percentiles=list(np.linspace(0, 1, 4))).reset_index()
            stat.columns = ["Quantile", var]
            stat[var] = stat[var].apply(lambda x: round(x, 2))
            source = ColumnDataSource(stat)
            cols = [TableColumn(field=x, title=x) for x in stat.columns]
            data_table = DataTable(source=source, columns=cols, width=config["total_width"], height=600)
            # Compute density
            fig_float_vars = figure(title="Densité de la variable : {}".format(var),
                                    tools="save",
                                    width=config["total_width"],
                                    height=800)
            fig_float_vars.xaxis.axis_label = 'x'
            fig_float_vars.yaxis.axis_label = 'Pr(x)'


            legend_it = []

            values = np.sort(values)
            if len(values) == 0:
                print("\tNo values for {}".format(var))
            else:
                min_, max_ = None, None
                try:
                    min_, max_ = np.percentile(values, q=[1, 99])
                except:
                    min_ = min_ or values.min()
                    max_ = max_ or values.max()

                hist, edges = np.histogram(values[(values > min_) & (values < max_)], density=True, bins=50)
                fig_float_vars.quad(top=hist,
                                    bottom=0,
                                    left=edges[:-1],
                                    right=edges[1:],
                                    line_color="#033649")

                try:

                    kwargs = {"y": [0, hist.max()],
                              "line_dash": "dashed",
                              "color": "red",
                              "line_width": 2}
                    q1, q3 = np.percentile(values, q=list(100*np.linspace(0, 1, 4))[1:-1])

                    fig_float_vars.circle(x=[q1, q1], **kwargs)
                    c = fig_float_vars.line(x=[q1, q1], **kwargs)
                    legend_it.append(("q1 : {}".format(q1), [c]))

                    kwargs["color"] = "green"
                    fig_float_vars.circle(x=[q1, q3], **kwargs)
                    c = fig_float_vars.line(x=[q3, q3], **kwargs)
                    legend_it.append(("q3 : {}".format(q3), [c]))
                except Exception as e:
                    print(e)

                legend = Legend(items=legend_it)
                legend.click_policy = "mute"
                fig_float_vars.add_layout(legend, 'below')

                try:
                    script, fig = components(row(data_table, fig_float_vars))

                    context["bokeh_scripts"].append(script)
                    context["float_vars"][var] = fig
                except:
                    pass

    all_vars = list(set(list(float_vars) + list(date_vars)))
    all_vars_encoder = {k: MyKBinsDiscretizer(n_bins=4, encode='ordinal', strategy="quantile") for k in all_vars}

    df_binned = df[all_vars].copy()
    for var in all_vars:
        print("Binning : {}".format(var))
        df_binned.loc[pd.isnull(df_binned[var]) , var] = np.nan
        df_binned.loc[:, var] = df_binned.loc[:, var].astype(float)
        df_binned[var] = all_vars_encoder[var].fit_transform(df_binned[var].values)

    fig = analyse_corr_2(df_binned[all_vars],
                       limit=len(df_binned),
                       plot=False).fig
    context["heatmap_full"] = to_html_string(fig)

    candidate_vars, _ = analyse_corr(df_binned[all_vars],
                                  limit=len(df_binned),
                                  plot=False)
    fig = analyse_corr_2(df_binned[candidate_vars],
                       limit=len(df_binned),
                       plot=False).fig
    context["candidate_vars"] = candidate_vars



    candidate_vars_dict = {}
    _dict = {"date": date_vars,
             "object": object_vars,
             "float": float_vars}
    for var_type, var_list in _dict.items():
        for var in var_list:
            if var in candidate_vars:
                candidate_vars_dict[var] = var_type
    """
    {'Date dépôt PC': 'date',
     'Date réitération': 'date',
     'Code postal': 'float',
     'Nb jours ouvrès : promesse - PC ': 'float',
     'Nb jours ouvrès : PC- Travaux ': 'float',
     'Nb jours ouvrès : Travaux - Livraison': 'float',
     'Nb de lots': 'float',
     'Nb de T5': 'float',
     'Nb de maisons': 'float',
     'Surface du terrain': 'float',
     "Nb d'étages": 'float',
     'Nb de niveaux de sous sol': 'float',
     'Nb de places de parkings': 'float',
     'Présence d’au moins un ascenseur': 'float',
     'Honoraires de vente (K€)': 'float',
     'CA TTC Social': 'float'}
    """

    context["candidate_vars_dict"] = candidate_vars_dict
    context["heatmap_sub"] = to_html_string(fig)



    #====================================== Output graphs to html   =======================================================#
    env_jinja = jinja2.Environment(loader=jinja2.FileSystemLoader(config["path_dir_template"]))
    html_string = env_jinja.get_template(u"templates/templateDataExploration.html").render(**context)
    # Save html
    with open("test.html", 'wb') as f:
        f.write(html_string.encode("utf-8"))



