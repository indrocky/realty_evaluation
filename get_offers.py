# web
from selectolax.parser import HTMLParser
from bs4 import BeautifulSoup

# other
import datetime
import time
import sqlite3

# analytics
import pandas as pd
import numpy as np


def get_offers(data, session, headers, site, offer_type, new, secondary):
    """
    Функция получения информации из объявлений.
    Возвращает список словарей с информицией из объявления
    """
    offers = []

    curr_idx, columns = current_idx()

    for key in data.keys():

        if 'single-page' in key:
            items = data[key]['data']['catalog']['items']

            for item in items:

                if item.get('id'):
                    offer = {key: value for key, value in zip(columns, [None] * len(columns))}
                    offer['offer_id'] = item['id']

                    if offer['offer_id'] not in curr_idx:

                        offer['add_time'] = str(datetime.datetime.now())
                        offer['model_participation'] = 'test'

                        offer['price'] = item['priceDetailed']['value']
                        offer['title'] = item['title'].strip()

                        try:
                            offer['city'] = item['geo']['geoReferences'][0]['content']
                        except (IndexError, KeyError):
                            pass

                        offer['address'] = item['geo']['formattedAddress']
                        offer['url_offer'] = site + item['urlPath']  # ссылка для перехода в карточку
                        offer['offer_date'] = str(datetime.datetime.fromtimestamp(item['sortTimeStamp'] / 1000))

                        if offer_type == secondary:
                            offer['offer_type'] = 'вторичка'

                        elif offer_type == new:
                            offer['offer_type'] = 'первичка'

                        else:
                            print('offer_type ERROR')
                            offer['offer_type'] = 'ERROR'

                        try:
                            offer['remote'] = item['geo']['geoReferences'][0]['after']
                        except (IndexError, KeyError):
                            pass
                        try:
                            offer['remote_time'] = item['geo']['geoReferences'][0]['afterWithIcon']['text']
                        except (IndexError, KeyError):
                            pass

                        offer, status = get_item_info(session, offer['url_offer'], offer, headers)
                        offers.append(offer)

                        if status != 200:
                            print("status ain't 200")
                            offers.pop()
                            break

    return offers


def get_item_info(session, url, offer: dict, headers):
    """
    Используется в функции get_offers
    Функция получения информации из карточки объявления
    """
    sleep_num = np.random.random() + np.random.randint(10, 15)
    time.sleep(sleep_num)
    unsuccessful_string = 'item card visit unsuccessful'

    cnt = 0
    item_response = unsuccessful_string

    while cnt < 5:
        cnt += 1

        try:
            item_response = session.get(url, headers=headers)
            break
        except Exception as e:
            print(unsuccessful_string, cnt, e.args)
            print(url)
            time.sleep(np.random.randint(3, 5))

    if item_response == unsuccessful_string:
        return offer, None

    elif item_response.status_code == 200:
        item_response.encoding = 'utf-8'
        item_html = item_response.text
        item_tree = HTMLParser(item_html)
        nodes = item_tree.css('li')

        for node in nodes:
            n_char = node.text().split(':')[0].strip()
            n_value = node.text().split(':')[-1].strip()
            if 'Количество комнат' in node.text():
                offer['room_num'] = n_value
            elif n_char == 'Общая площадь':
                offer['square'] = n_value
            elif n_char == 'Площадь кухни':
                offer['kitchen_area'] = n_value
            elif n_char == 'Жилая площадь':
                offer['living_area'] = n_value
            elif n_char == 'Этаж':
                offer['floor'] = n_value
            elif n_char == 'Балкон или лоджия':
                offer['balcony'] = n_value
            elif n_char == 'Тип комнат':
                offer['room_type'] = n_value
            elif n_char == 'Высота потолков':
                offer['ceiling_height'] = n_value
            elif n_char == 'Санузел':
                offer['bathroom'] = n_value
            elif n_char == 'Окна':
                offer['windows'] = n_value
            elif n_char == 'Ремонт' or n_char == 'Отделка':
                offer['renovation'] = n_value
            elif n_char == 'Мебель':
                offer['furniture'] = n_value
            elif n_char == 'Способ продажи':
                offer['sell_type'] = n_value
            elif n_char == 'Вид сделки':
                offer['deal_type'] = n_value
            elif n_char == 'Тип дома':
                offer['house_type'] = n_value
            elif n_char == 'Год постройки' or n_char == 'Срок сдачи':
                offer['built_in'] = n_value
            elif n_char == 'Этажей в доме':
                offer['total_floor'] = n_value
            elif n_char == 'Пассажирский лифт':
                offer['passenger_lift'] = n_value
            elif n_char == 'Грузовой лифт':
                offer['heavy_lift'] = n_value
            elif n_char == 'В доме':
                offer['house_attributes'] = n_value
            elif n_char == 'Двор':
                offer['yard'] = n_value
            elif n_char == 'Парковка':
                offer['parking'] = n_value

        soup = BeautifulSoup(item_html, "html.parser")
        all_divs = soup.findAll('div')

        for d in all_divs:
            if 'data-map-lat' in d.attrs.keys():
                offer['map_lat'] = d['data-map-lat']
                offer['map_lon'] = d['data-map-lon']

        return offer, item_response.status_code

    else:
        print('get_offers.py, something wrong here', item_response.status_code)
        print(url)

        return offer, item_response.status_code


def current_idx():
    """
    Используется в функции get_offers
    Функция получения уже записанных индексов объявлений в базу данных и существующих колонок
    Возвращает списоки индексов объявлений и названий колонок
    """

    with sqlite3.connect('data_base/realty.db') as connection:
        cursor = connection.cursor()
        cursor.execute('select offer_id from offers')
        current_idx = cursor.fetchall()
        current_idx = [i[0] for i in current_idx]

        columns = pd.read_sql('select * from offers where offer_id is null', connection).columns.tolist()[1:]

        return current_idx, columns
