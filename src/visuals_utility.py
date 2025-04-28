from colorama import Style, Fore, Back
import logger_setup
import pandas as pd
from matplotlib_venn import venn2
from conf.config import PATH_OUT_VISUALS, MODEL_VERSION

# importing visualisation libraries & stylesheets
import matplotlib.pyplot as plt
from conf.config import MPL_STYLE_FILE
plt.style.use(MPL_STYLE_FILE)

class ColourStyling(object):
    blk = Style.BRIGHT + Fore.BLACK
    gld = Style.BRIGHT + Fore.YELLOW
    grn = Style.BRIGHT + Fore.GREEN
    red = Style.BRIGHT + Fore.RED
    blu = Style.BRIGHT + Fore.BLUE
    mgt = Style.BRIGHT + Fore.MAGENTA
    res = Style.RESET_ALL

custColour = ColourStyling()

# function to render colour coded print statements
def beautify(str_to_print: str, format_type: int = 0) -> str:
    if format_type == 0:
        return custColour.mgt + str_to_print + custColour.res
    if format_type == 1:
        return custColour.grn + str_to_print + custColour.res
    if format_type == 2:
        return custColour.gld + str_to_print + custColour.res
    if format_type == 3:
        return custColour.red + str_to_print + custColour.res

def plot_line(list_of_df: list, list_of_labels: list, x_col, y_col, color='teal', figsize: tuple = (8, 6), dpi: int = 130):
    logger_setup.logger.debug("START ...")
    if list_of_labels is None:
        labels = [f'Line {i + 1}' for i in range(len(list_of_df))]

    for idx, df in enumerate(list_of_df):
        plt.plot(df[x_col], df[y_col], label=list_of_labels[idx], marker='o')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Multiple Line Plots')
    plt.legend()

    # Saving the plot as an image file
    plt.savefig(f'{PATH_OUT_VISUALS}optuna_model_perf_{MODEL_VERSION}.png')
    logger_setup.logger.debug("... FINISH")

def plot_filled_values_percent(df: pd.DataFrame, color='teal', figsize: tuple = (8, 6), dpi: int = 130):
    logger_setup.logger.info('START ...')
    filled_values_percent = (df.notnull().sum() / len(df) * 100).sort_values()
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    # Create bars representing total values (set to 100)
    axes.barh(filled_values_percent.index, [100] * len(df.columns), color='#f5f5f5')
    # Create bars representing filled values
    axes.barh(filled_values_percent.index, filled_values_percent, color='turquoise')

    axes.set_xlim([0, 100])
    axes.set_xlabel('Percentage (%) filled')
    axes.set_title('Percentage of filled values in each column')
    plt.show()
    logger_setup.logger.info('... FINISH')

def plot_cat_col_cardinality(df: pd.DataFrame, color='turquoise', height=0.75,
                             figsize: tuple = (6, 9), dpi: int = 150):
    logger_setup.logger.info('START ...')
    cardinality = df.nunique().sort_values(ascending=False)
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    axes.barh(cardinality.index, cardinality, color=color, height=height)
    for i in range(len(df.columns)):
        axes.text(cardinality.iloc[i] + 0.5, i,
                  f'{str(cardinality.iloc[i])} among {df[cardinality.index[i]].count()}', va='center',
                  fontsize=7)
    axes.set_xlim([0, cardinality.iloc[0] + 5])
    axes.set_xlabel('Unique values (aka cardinality) among total non null values')
    axes.set_title('Cardinality of categorical columns')
    plt.show()
    logger_setup.logger.info('... FINISH')

def plot_venn_diagram(df1, df1_display_name, df2, df2_display_name, join_column, join_column_display_name, figsize: tuple = (8, 6), dpi: int = 150):
    logger_setup.logger.info('START ...')
    # Convert the joining column to a set for each dataframe
    set1 = set(df1[join_column])
    set2 = set(df2[join_column])

    common_values = set1.intersection(set2)
    # Create the venn diagram
    fig, axes = plt.subplots(figsize=figsize, dpi=dpi)
    venn = venn2([set1, set2], (df1_display_name, df2_display_name))

    # Display the plot
    plt.title(f"Venn Diagram for {join_column_display_name}")
    plt.show()
    logger_setup.logger.info('... FINISH')
    return list(common_values)

def plot_cardinality(cardinality_df, n_cat_threshold, threshold_used='ABS', type_of_cols='all', figsize=(10, 6)):
    """
    Plots cardinality (percentage) of columns as a stacked bar plot.

    Parameters:
        cardinality_df (DataFrame): DataFrame has four columns in this specific order: column name, not-null percentage, null percentage, and unique percentage.
        n_cat_threshold (float): Threshold set to identify a possible categorical feature and mark it on the plot.
        type_of_cols (str): Description for the column type for plot title (e.g. 'categorical').
        threshold_used (str): Threshold type used for identifying categorical features (e.g. 'ABS').
        figsize (tuple): Size of the plot. Defaults to (10, 6).
    """

    stack_colours = ['#deffd4', '#ffffff']

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cardinality_df.iloc[:,:-1].plot.bar(
        x=cardinality_df.columns[0],
        stacked=True,
        ax=ax,
        linewidth=0.75,
        edgecolor="gray",
        color=stack_colours
    )
    ax.invert_xaxis()
    ax.set_xlabel('Column names')
    ax.set_ylabel('Percentage of rows')
    ax.set_title(f'Cardinality plot of {type_of_cols} columns')

    # Add a black dash for each bar to signify 'unique_pct' values
    for i, col_name in enumerate(cardinality_df.iloc[:,0]):
        unique_value = cardinality_df.loc[cardinality_df.iloc[:, 0] == col_name, cardinality_df.columns[-1]].values[0]
        ax.plot(i, unique_value, '_', markeredgecolor = 'black', markersize=10, markeredgewidth=1, label=('unique_pct' if i == 0 else None))

    if threshold_used == 'PCT':
        ax.axhline(y=n, color='red', linestyle='-', linewidth=1, alpha=0.8, label=f'Threshold line at {n_cat_threshold}')

    # anchoring the legend box lower left corner to below X/Y coordinates scaled 0-to-1
    plt.legend(bbox_to_anchor=(1.0, 0))
    plt.show()


def plot_numerical_distribution(df, features):
    if features is None or len(features) == 0:
        return

    # Calculate the number of rows and columns needed
    n_features = len(features)
    n_cols = 6  # Maximum cols to have in the plot grid
    n_rows = (n_features + n_cols - 1) // n_cols

    # Make axs 2D if it's 1D
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    feat_idx = 0
    for i in range(n_rows):
        fig, axs = plt.subplots(2, cols, gridspec_kw={"height_ratios": (0.7, 0.3)}, figsize=(4*n_cols, 4))
        for j in range(n_cols):
            axs_hist = axs[0, j]
            axs_box = axs[1, j]
            if feat_idx < len(features):
                # Plot hist chart for the current feature
                current_feature = features[feat_idx]
                axs_hist.hist(df[current_feature], color='lightgray', edgecolor='gray', linewidth=0.5, bins=50)
                axs_hist.set_title(f'Plots for {current_feature}', fontsize=10)
                axs_hist.spines['top'].set_visible(False)
                axs_hist.spines['right'].set_visible(False)

                axs_box.boxplot(
                df[current_feature],
                vert=False,
                widths=0.7,
                patch_artist=True,
                medianprops={
                    'color': 'black'
                },
                flierprops={
                    'marker': 'o',
                    'markerfacecolor': 'gray',
                    'markersize': 2
                },
                whiskerprops={
                    'linewidth': 0.5
                },
                boxprops={
                    'facecolor': 'lightgray',
                    'color': 'gray',
                    'linewidth': 1
                },
                capprops={
                    'linewidth': 1
                }
                )

                axs_box.set(yticks=[])
                axs_box.spines['left'].set_visible(False)
                axs_box.spines['right'].set_visible(False)
                axs_box.spines['top'].set_visible(False)

                feat_idx += 1
            else:
                # Hide empty subplots
                axs_hist.set_visible(False)
                axs_box.set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(df, features):
    if features is None or len(features) == 0:
        return

    # Calculate the number of rows and columns needed
    n_features = len(features)
    n_cols = 6  # Maximum cols to have in the plot grid
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create a figure with a proper subplot grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs = axs.ravel() # Flatten the array for easier indexing

    for idx, feature in enumerate(features):
        value_counts = df[feature].value_counts().sort_index()
        axs[idx].bar(range(len(value_counts)), value_counts.values, color='lightgray', edgecolor='gray', linewidth=0.5)

        # Set both the tick positions and labels
        axs[idx].set_xticks(range(len(value_counts)))  # Set tick positions
        axs[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')  # Set tick labels
        axs[idx].set_title(f'Distribution of {feature}', fontsize=10)

    # Hide empty subplots
    for idx in range(len(features), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_relationship_to_target(df, features, target, trend_type=None):
    if features is None or len(features) == 0:
        return

    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axs = axs.ravel()  # Flatten the array for easier indexing

    for idx, feature in enumerate(features):
        # Group data by feature
        grouped_data = [group[target].values for name, group in df.groupby(feature)]

        # Create box plot
        axs[idx].boxplot(
            grouped_data,
            patch_artist=True,
            medianprops={
                'color': 'black'
            },
            flierprops={
                'marker': 'o',
                'markerfacecolor': 'gray',
                'markersize': 2
            },
            whiskerprops={
                'linewidth': 1
            },
            boxprops={
                'facecolor': 'lightgray',
                'color': 'gray',
                'linewidth': 1
            },
            capprops={
                'linewidth': 1
            }
        )

        # Set x-ticks with feature categories
        categories = sorted(df[feature].unique())
        axs[idx].set_xticklabels(categories, rotation=45, ha='right')
        axs[idx].set_title(f'Distribution of {target} by {feature}', fontsize=10)

        # Add trend line if specified
        if trend_type is not None:
            axs_twin_y = axs[idx].twinx()

            if trend_type == 'mean':
                trend_values = df.groupby(feature)[target].mean()
            elif trend_type == 'median':
                trend_values = df.groupby(feature)[target].median()

            # Plot trend line
            axs_twin_y.plot(
                range(1, len(categories) + 1),
                trend_values.values,
                color='red',
                marker='o',
                markersize=3,
                linewidth=1,
                alpha=0.6
            )

            # Set trend line axis properties
            axs_twin_y.tick_params(axis='y', colors='red')

    # Hide empty subplots
    for idx in range(len(features), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()