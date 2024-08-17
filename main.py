import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def neighbourhood_distribution_bar_plot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    counts = df['neighbourhood_group'].value_counts().sort_values()

    counts.plot(kind='bar', color=['#69A197', '#F39237', '#D63230', '#22577A', '#242038'])

    plt.title('Listings of different neighbourhood groups')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Number of Listings')
    plt.xticks(rotation=0)

    for i, count in enumerate(counts.values):
        plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=12)

    plt.savefig('1.png')
    plt.show()


def price_distribution_plot(df: pd.DataFrame):
    box_plot = df.boxplot(column='price', by='neighbourhood_group', sym='*', grid=False, patch_artist=True, figsize=(10, 6))
    colors = ['#69A197', '#F39237', '#D63230', '#22577A', '#242038']

    for patch, color in zip(box_plot.artists, colors):
        patch.set_facecolor(color)

    plt.title('Different neighbourhood prices')
    plt.suptitle('')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Price')

    plt.savefig('2.png')
    plt.show()


def grouped_bar_plot(df: pd.DataFrame):
    grouped_mean = df.groupby(['neighbourhood_group', 'room_type']).agg({
        'availability_365': 'mean'
    }).unstack()
    grouped_std = df.groupby(['neighbourhood_group', 'room_type']).agg({
        'availability_365': 'std'
    }).unstack()

    grouped_mean.plot(kind='bar', yerr=grouped_std)

    plt.title('Average availability for each room type across the neighborhoods')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Average availability')
    plt.xticks(rotation=0)
    plt.legend(labels=['Entire home/apt', 'Private room', 'Shared room'])

    plt.savefig('3.png')
    plt.show()


def scatter_plot(df: pd.DataFrame):
    colors = {'Entire home/apt': '#001B2E', 'Private room': '#7F95D1', 'Shared room': '#EA638C'}
    lines = {'Entire home/apt': '--', 'Private room': '-.', 'Shared room': ':'}

    room_types = df['room_type'].unique()

    plt.figure(figsize=(10, 6))

    for room_type in room_types:
        listings = df[df['room_type'] == room_type]
        plt.scatter(x=listings['price'], y=listings['number_of_reviews'], alpha=0.7, color=colors.get(room_type), label=room_type)

        prices = listings['price'].values
        reviews = listings['number_of_reviews'].values

        gradient, intercept_point = np.polyfit(prices, reviews, deg=1)
        plt.plot(prices, gradient * prices + intercept_point, color=colors.get(room_type), linestyle=lines.get(room_type))

    plt.title('Scatter plot price vs number of reviews by room type')
    plt.xlabel('Price')
    plt.ylabel('Number of reviews')
    plt.legend(title='Room types')
    plt.savefig('4.png')
    plt.show()


def line_plot(df: pd.DataFrame):
    copy = df.copy()
    copy['last_review_date'] = pd.to_datetime(copy['last_review'])

    copy = copy.sort_values(['neighbourhood_group', 'last_review_date'])

    copy['smoothed_reviews'] = copy.groupby('neighbourhood_group')['number_of_reviews'].rolling(window=3).mean().reset_index(level=0, drop=True)
    copy = copy.groupby(['neighbourhood_group', 'last_review_date'])['smoothed_reviews'].mean().reset_index()

    pivot = pd.pivot(copy, index='last_review_date', columns='neighbourhood_group', values='smoothed_reviews')

    plt.figure(figsize=(12, 6))

    colors = {'Bronx': '#69A197', 'Manhattan': '#F39237', 'Brooklyn': '#D63230', 'Queens': '#22577A', 'Staten Island': '#242038'}

    for column in pivot.columns:
        plt.plot(pivot.index, pivot[column], c=colors.get(column), label=column)

    plt.title('Number of reviews over time for each neighbourhood group')
    plt.xlabel('Last review date')
    plt.ylabel('Average number of reviews')
    plt.legend(title='Neighbourhood group')

    plt.savefig('5.png')
    plt.show()


def heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index='neighbourhood_group', columns='price', values='availability_365', aggfunc='mean')
    pivot.fillna(0, inplace=True)

    plt.figure(figsize=(10,8))

    plt.imshow(pivot, cmap='YlGnBu', aspect='auto')
    plt.colorbar(label='Average Availability')

    plt.xlabel('Price')
    plt.ylabel('Neighbourhood group')
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title('Relationship between price and availability_365 across different neighborhoods')

    plt.savefig('6.png')
    plt.show()


def room_type_review_count(df: pd.DataFrame):
    pivot = pd.pivot_table(df, index='neighbourhood_group', columns='room_type', values='number_of_reviews', aggfunc='count')

    pivot.plot.bar(stacked=True)

    plt.title('Reviews for each room type across neighbourhood groups')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.legend(labels=['Entire home/apt', 'Private room', 'Shared room'])

    plt.savefig('7.png')
    plt.show()


data = pd.read_csv('AB_NYC_2019.csv')

# 1
neighbourhood_distribution_bar_plot(data)

# 2
price_distribution_plot(data)

# 3
grouped_bar_plot(data)

# 4
scatter_plot(data)

# 5
line_plot(data)

# 6
heatmap(data)

# 7
room_type_review_count(data)
