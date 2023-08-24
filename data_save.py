# other
import sqlite3

# analytics
import pandas as pd


def get_df(offers_data):
    """
    Функция получения дата-фрейма из списка словарей.
    """

    df = pd.DataFrame(offers_data)
    return df


def data_save_csv(df):
    """
    Функция сохранения новых объявлений в .csv файл
    """

    df_check = pd.read_csv('data_base/realty_offers.csv')
    df_to_add = df[~df['offer_id'].isin(df_check['offer_id'].tolist())]
    df_check = pd.concat([df_check, df_to_add])
    df_check.to_csv('data_base/realty_offers.csv', index=False)
    print(f'{df_to_add.shape[0]} объявлений добавлено в csv файл')

    return df_to_add.shape[0]


def data_load_to_db(df):
    """
    Функцция загрузки новых объявлений из дата фрейма в базу данных
    """

    for column in df.columns:
        try:
            df[column] = df[column].str.strip()
        except AttributeError:
            continue

    with sqlite3.connect('data_base/realty.db') as connection:
        cursor = connection.cursor()
        i = 0

        for offer in df.values:
            offer_id = offer[0]

            cursor.execute('select offer_id from offers where offer_id = (?)', (offer_id,))
            result = cursor.fetchone()

            if result is None:
                cursor.execute("""
                    insert into offers (
                       'offer_id', 'price', 'title', 'city', 'address', 'url_offer',
                       'offer_date', 'remote', 'remote_time', 'renovation', 'furniture',
                       'room_num', 'square', 'kitchen_area', 'living_area', 'floor', 'balcony',
                       'room_type', 'ceiling_height', 'bathroom', 'sell_type', 'house_type',
                       'built_in', 'total_floor', 'passenger_lift', 'windows', 'heavy_lift',
                       'yard', 'parking', 'deal_type', 'house_attributes', 'offer_type',
                       'add_time', 'model_participation', 'map_lat', 'map_lon')
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, offer)
                connection.commit()

                i += 1

    print(f'{i} объявлений добавлено в базу данных')
