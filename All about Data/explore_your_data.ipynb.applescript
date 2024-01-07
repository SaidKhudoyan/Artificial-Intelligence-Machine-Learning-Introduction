{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e45c308d",
   "metadata": {},
   "source": [
    "<div style=\"text-align: center;\">\n",
    "    <h1 style=\"color: #3498db;\">Artificial Intelligence & Machine Learning</h1>\n",
    "    <h2 style=\"color: #3498db;\">Part 1: Exploratory Data Analysis</h2>\n",
    "</div>\n",
    "\n",
    "-------------------------------------------------------------"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34da13e1",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    <b>Authors:</b> K. Said<br>\n",
    "    <b>Date:</b> 08-09-2023\n",
    "</div>\n",
    "\n",
    "<div style=\"background-color: #e6e6e6; padding: 10px; border-radius: 5px; margin-top: 10px;\">\n",
    "    <p>This notebook is part of the \"Artificial Intelligence & Machine Learning\" lecture material. The following copyright statement applies to all contents and code within this file.</p>\n",
    "    <b>Copyright statement:</b>\n",
    "    <p>This material, no matter whether in printed or electronic form, may be used for personal and non-commercial educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether in printed or in electronic form, requires explicit prior acceptance of the authors and lecturers.</p>\n",
    "</div>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a40dcc66",
   "metadata": {},
   "source": [
    "<h1 style=\"color:rgb(0,120,170)\">Introduction</h1>\n",
    "\n",
    "-----------------------------------------------\n",
    "\n",
    "<h2 style=\"color:rgb(0,120,0)\">What you have learned so far</h2>\n",
    "\n",
    "--------------------------------------------------------------------\n",
    "\n",
    "So far we have covered the basics of Machine Learning, including its applications and landscape. \n",
    "Afterwards we delved into the crucial topic of datasets and their types. Moreover, we've taken a significant step by selecting our own dataset, setting the stage for practical applications ahead. \n",
    "\n",
    "\n",
    "<h2 style=\"color:rgb(0,120,0)\">Why data analysis for ML?</h2>\n",
    "\n",
    "------------------------------------------------------\n",
    "Well for that, imagine for a second you want to start a machine learning project, where you are provided with tabular data. By looking at the raw table, one might conclude that this dataset just fits perfectly fine for the model, but when training the model, the accuracy might end up at 60% or worse, even if the model is perfectly fine, the accuracy also depends on the quality of the dataset.\n",
    "And for this reason we first try to analyse our dataset. Are there any missing values or values in the wrong format? Are there any patterns that can only be seen when visualizing the dataset? Which features should we use and how do the features depend on each other?\n",
    "Those are only a few questions that can be asked when trying to analyse the data and can have a tremendous impact on the prediction of your model.\n",
    "\n",
    "<h2 style=\"color:rgb(0,120,0)\">Our Task</h2>\n",
    "\n",
    "-------------------------------------------------\n",
    "\n",
    "In this notebook your task is therefore to explore your chosen dataset from the previous task and try to understand the relations between each feature, look for specific patterns and missing values that might lead to problems, but also have a glimpse at the distribution and insights that may be interesting for the model implementation.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0dbbc410",
   "metadata": {},
   "source": [
    "<h1 style=\"color:rgb(0,120,170)\">Data Analysis - Example</h1>\n",
    "\n",
    "-----------------------------------------------\n",
    "\n",
    "Now we can finally get started with your first more challenging exercise, data exploration. Here we will try to make use of some of the most common methods for exploring your dataset. In Order to make it easier for you to get started, we will first start by showing you some data analysis examples based on the penguins dataset.\n",
    "\n",
    "The penguins dataset was created by [Dr.Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php ) and the [Palmer Station, Antarctica LTER](https://pallter.marine.rutgers.edu/) , a member of the [Long Term Ecological Research Network.](https://lternet.edu/)\n",
    "\n",
    "The dataset is often used by people when starting with their very first ML and data analysis project.It contains information about various attributes of penguin species, making it a useful dataset for classification and data exploration exercises. \n",
    "\n",
    "For more information [here the link to the dataset](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c1bd28d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let us first import some packages we need for the visualizations\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "192c74d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we load our toy dataset \n",
    "penguins = sns.load_dataset(\"penguins\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5da01889",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: rgba(255, 165, 0, 0.6); border: 1px solid grey; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">Display the dataset</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(255, 204, 153, 0.6); padding: 10px; border-radius: 5px;\">\n",
    "    Before starting with any specific plots, we first want to get an overview of our dataset. This can be done easily by using e.g. pandas library, which not only allows us to get the overview, but also some statistical insights.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42f5bdb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# In order to plot multiple things in one cell, we use the display method of pandas\n",
    "display(penguins, penguins.describe(include='all'), penguins.dtypes)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "780d96b1",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: rgba(255, 165, 0, 0.6); border: 1px solid grey; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">Any missing values?</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(255, 204, 153, 0.6); padding: 10px; border-radius: 5px;\">\n",
    "    Well, from above plot we can clearly see that both dataframes contain NaN values. Quite interesting, right? In order to get a better understanding, we will try to plot the amount of missing values per each feature next.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "598172c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the percentage of missing values for each feature\n",
    "nan_percentage = (penguins.isna().mean() * 100).sort_values(ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=nan_percentage.index, y=nan_percentage.values, palette=\"viridis\")\n",
    "plt.title(\"Percentage of Missing Values by Feature in Penguins Dataset\", fontsize=16)\n",
    "plt.ylabel(\"Percentage of Missing Values\", fontsize=14)\n",
    "plt.xlabel(\"Features\", fontsize=14)\n",
    "plt.xticks(rotation=45, ha=\"right\", fontsize=12)\n",
    "plt.yticks(fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5145dc45",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    3% of the data in column \"sex\" are missing and around 0.5% in the other columns are missing. When a dataset has missing values, this can often happen due to collection errors, incomplete records, or sensor malfunctions during data gathering\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8525004c",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: rgba(255, 165, 0, 0.6); border: 1px solid grey; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">What is the distribution of each species?</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(255, 204, 153, 0.6); padding: 10px; border-radius: 5px;\">\n",
    "    Alright, we know that the dataset contains missing values. But what about the ratio of the different species? In order to find that out, we will now try to look at a simple pie-chart.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f602608f",
   "metadata": {},
   "outputs": [],
   "source": [
    "species_counts = penguins[\"species\"].value_counts()\n",
    "\n",
    "plt.figure(figsize=(8, 8))\n",
    "plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)\n",
    "plt.title(\"Distribution of Penguin Species\")\n",
    "plt.axis('equal')\n",
    "\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "93e22618",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    Quite interesting, so we have 3 different species, with the majority being of type <a href=\"https://en.wikipedia.org/wiki/Ad%C3%A9lie_penguin\" style=\"color: blue; text-decoration: underline;\" target=\"_blank\">Adelie</a>.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "975a1f83",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: rgba(255, 165, 0, 0.6); border: 1px solid grey; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">What is the distribution of each species?</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(255, 204, 153, 0.6); padding: 10px; border-radius: 5px;\">\n",
    "   Now out of curiosity, we want to plot a boxplot of the bill_length distribution of each species. For the first boxplot we use seaborn, for the second one we use the more interactive plotly library.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd62de58",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Seaborn box plot for \"bill_length_mm\"\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.boxplot(x=\"species\", y=\"bill_length_mm\", data=penguins, palette=\"viridis\")\n",
    "plt.title(\"Seaborn - bill_length_mm Distribution by Penguin Species\")\n",
    "plt.xlabel(\"Penguin Species\")\n",
    "plt.ylabel(\"bill_length_mm\")\n",
    "plt.show()\n",
    "\n",
    "# Plotly box plot for \"bill_length_mm\"\n",
    "fig = px.box(penguins, x=\"species\", y=\"bill_length_mm\", title=\"Plotly - bill_length_mm Distribution by Penguin Species\")\n",
    "fig.update_layout(showlegend=False)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "417fa5e6",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    Well, with the above plots, we can clearly see the min, max, and median values. We also see that the Adelie species has a much shorter bill_length, this might come in handy when we choose the features to train our models with.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65cbf1f8",
   "metadata": {},
   "source": [
    "\n",
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: rgba(255, 165, 0, 0.6); border: 1px solid grey; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">Any correlation between the features?</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(255, 204, 153, 0.6); padding: 10px; border-radius: 5px;\">\n",
    "    Now we want to see the correlation between the features, which may be useful for the preprocessing and model selection part. For this purpose we will first create a scatter-plot, followed by a correlation heatmap.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ad9cde9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scatter plot with all features using Seaborn and \"Sex\" as the hue\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.pairplot(data=penguins, hue=\"species\")\n",
    "plt.suptitle(\"Scatter Plot of Penguins Features (All Variables)\", fontsize=16)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ca83690",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    Having a closer look, we can see that it is much easier for us to separate the blue (Adelie) species from the green (Gentoo) species than the blue from the orange (Chinstrap).\n",
    "</div>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "530168ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation matrix for all features, also called Heatmap\n",
    "corr_matrix = penguins.corr(numeric_only=True)\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap=\"coolwarm\", fmt=\".2f\", linewidths=0.5)\n",
    "plt.title(\"Heatmap of Correlation Matrix for Penguin Features (All Variables)\", fontsize=16)\n",
    "plt.xticks(fontsize=12)\n",
    "plt.yticks(fontsize=12)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e393f7ae",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f2f2f2; padding: 10px; border-radius: 5px;\">\n",
    "    We see quite a high correlation between body_mass_g and flipper_length_mm (0.87). bill_length and bill_depth on the other hand have a rather very low correlation.\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f95fec8",
   "metadata": {},
   "source": [
    "<h1 style=\"color:rgb(0,120,170)\">Data Analysis - Your turn</h1>\n",
    "\n",
    "-----------------------------------------------\n",
    "\n",
    "Now it's your turn to analyse your chosen dataset from the previous tasks. There are no limits, you can just reuse some parts of the previous section, ideas from the info button or try out new things and implement them yourself (which we recommend). \n",
    "\n",
    "First load your dataset and try to visualize it in a similar manner as above. You can also play around with different libraries (mentioned in \"Useful Ressources\" and try to create interactive plots.\n",
    "\n",
    "And just as a last sitenote, don't try to make as many plots as possible. Instead think to yourself, what information do I want to get from the dataset, what do I want to analyse? How will this specific plot help me gain useful insights of the data and how can I use this insights for the next step, preprocessing?\n",
    "\n",
    "Other than that, just feel free to explore your dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "763ae4e1",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: #3498db; border: 1px solid #3498db; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">Info</summary>\n",
    "\n",
    "<div style=\"background-color: #E6F7FF; padding: 10px; border-radius: 5px;\">\n",
    "    After watching the lectures, you might feel a bit lost, as there a tons of ways to analyse and visualize things. In order to make it easier for you to know which methods to apply, here a short overview of what you can do to analyse your dataset:\n",
    "\n",
    "   - **5-number-summary**\n",
    "     - Minimum\n",
    "     - Maximum\n",
    "     - Lower and Upper quantile\n",
    "     - Median\n",
    "\n",
    "   - **Mean & Standard Deviation**\n",
    "    \n",
    "   - **Investigate Outliers**\n",
    "\n",
    "   - **Plots**\n",
    "     - Violin-Plot\n",
    "     - Box-plot\n",
    "     - Histogram\n",
    "     - Bar charts\n",
    "     - Pie charts\n",
    "   - **Correlation Matrix**\n",
    "\n",
    "Of course this is only a small fraction of all available methods, therefore you might want to have a look at the \"Useful Resources\" to get some more inspiration.\n",
    "\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ec12253",
   "metadata": {},
   "source": [
    "<details>\n",
    "<summary style=\"font-size: larger; color: white; background-color: darkgreen; border: 1px solid darkgreen; padding: 5px 15px; border-radius: 8px; cursor: pointer;\">Useful Ressources</summary>\n",
    "\n",
    "<div style=\"background-color: rgba(0, 128, 0, 0.2); padding: 10px; border-radius: 8px; margin-top: 10px;\">\n",
    "    - <a href=\"https://matplotlib.org/stable/contents.html\">Matplotlib</a>: A popular plotting library for creating visualizations.\n",
    "    <br><br>\n",
    "    - <a href=\"https://seaborn.pydata.org/\">Seaborn</a>: A data visualization library based on Matplotlib, providing a high-level interface for creating informative and attractive statistical graphics.\n",
    "    <br><br>\n",
    "    - <a href=\"https://plotly.com/python/\">Plotly</a>: A library for creating interactive plots and dashboards. It supports a wide range of chart types and is great for sharing data visualizations online.\n",
    "    <br><br>\n",
    "    - <a href=\"https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html\">Pandas Plotting</a>: Pandas has built-in plotting capabilities that allow you to create basic plots directly from DataFrames.\n",
    "    <br><br>\n",
    "    - <a href=\"https://bokeh.org/\">Bokeh</a>: Bokeh is a library for creating interactive, web-ready visualizations. It's well-suited for building interactive dashboards.\n",
    "    <br><br>\n",
    "    - <a href=\"https://altair-viz.github.io/\">Altair</a>: Altair is a declarative statistical visualization library that is especially useful for creating complex, layered visualizations.\n",
    "</div>\n",
    "</details>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b82b7977",
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: Explore your dataset, find formating errors, missing values and much more\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "baaf6719",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c0a0304",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c64d8ca",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0267626a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5966557",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41e9223b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
