import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import sqlite3
import re
import requests
import warnings

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats

from config import token, chat_id

warnings.filterwarnings("ignore")


def get_data() -> 'DataFrame':
    """
    Функция для получения датафрейма с данными по квартирам
    """
    with sqlite3.connect(r'C:\Users\dmaryanovskiy\PycharmProjects\parsing\data_base\realty.db') as connection:
        query = 'select * from offers'
        df = pd.read_sql(query, connection)
        df_to_check = df[df['model_participation'] == 'test']
        print('------------------------ Checking empty values in test data ------------------------')
        df_empty_values = df_to_check.isna().sum()
        print(df_empty_values)
        if df_empty_values.max() == df_to_check.shape[0]:
            print('------------------------ Empty values check ------------------------')
            print('MIGHT BE AN ERROR IN PARSING ALGORITHM')

        return df


def data_formatting(data: 'DataFrame', show_plots: bool) -> 'DataFrame':
    """
    Функция предобработки данных из БД. Возвращает ДФ, на с корректно заполненными колонками с которыми можено работать дальше
    """

    def converting_to_numeric(x: str) -> float:
        """
        Функция преобразования строковых заначений в таблице в значения с плавающей точкой
        """
        if x is not None:
            var = re.search('[\d.,]+', x)
            if var is not None:
                return float(var[0].replace(',', '.'))

    def remote_m(x: str) -> float:
        """
        Функция преобразования строкового значения удаленности в значение с плавающей точкой
        """
        if re.search('\sм', x) is not None:
            return float(re.search('\d+', x)[0])
        elif re.search('км', x) is not None:
            return round(float(x.split()[0].replace(',', '.')) * 1000, 2)
        else:
            return 'Error'

    def ceiling_convert_to_numeric(x: str) -> float:
        """
        Функция преобразования строкового значения высоты потолков в значение с плавающей точкой
        """
        x = converting_to_numeric(x)
        if x is not None:
            return float(str(x * 100).replace('.', '')[:3])

    def extract_year_build(x: str) -> int:
        """
        Функция преобразования строкового значения года постройки в целочисленное значение
        """
        if x == 'сдан':
            return datetime.datetime.now().year
        elif x is not None:
            try:
                year = int(re.search('\d{4}', x)[0])
            except:
                year = 2000 + int(re.search('\d{2}', x)[0])
            if year >= 2050:
                year = int(str(year)[-2:]) + 2000
            return year
        else:
            return 0

    pre_shape = data.shape

    data.replace('', None, inplace=True)

    data['n_square'] = data['square'].apply(converting_to_numeric)
    data['m2_price'] = round(data['price'] / data['n_square'], 2)
    data['n_room_num'] = data['room_num'].apply(converting_to_numeric)
    data['n_built_in'] = data['built_in'].apply(extract_year_build)
    data['n_ceiling'] = data['ceiling_height'].apply(ceiling_convert_to_numeric)
    data['n_ceiling'].fillna(data['n_ceiling'].median(), inplace=True)

    data = data[
        (~data['remote'].isna()) &
        (data['remote_time'].isin(['до\xa05\xa0мин.', '6–10\xa0мин.', '11–15\xa0мин.', '16–20\xa0мин.'])) &
        (data['n_room_num'].isin([2, 3])) &
        ((~data['renovation'].isna()) & (data['renovation'] != 'Ремонт и строительство')) &
        (data['house_type'].isin(['кирпичный', 'монолитно-кирпичный', 'монолитный'])) &
        (data['n_built_in'] >= 2000) &
        (data['floor'].str.contains('из')) &
        (~data['title'].isna()) &
        (data['n_ceiling'].between(270, 400))
        ]

    data.loc[data['city'].str.contains('Невского'), 'city'] = 'Площадь А. Невского'
    data.loc[data['city'].str.contains('Технологический'), 'city'] = 'Технологический ин-т'

    data['n_kitchen_area'] = data['kitchen_area'].apply(converting_to_numeric)
    data['n_living_area'] = data['living_area'].apply(converting_to_numeric)
    kitchen_share = (data['n_kitchen_area'] / data['n_square']).median()
    living_share = (data['n_living_area'] / data['n_square']).median()
    data['n_kitchen_area'].fillna(data['n_square'] * kitchen_share, inplace=True)
    data['n_living_area'].fillna(data['n_square'] * living_share, inplace=True)

    data['remote_m'] = data['remote'].apply(remote_m)
    data['n_floor'] = data['floor'].apply(converting_to_numeric)

    data['balcony'].fillna('без балкона', inplace=True)
    data['has_concierge'] = data['house_attributes'].apply(lambda x: 1 if type(x) == str and 'консьерж' in x else 0)

    data['apartments'] = 0
    data.loc[data['title'].str.contains('апартаменты', case=False), 'apartments'] = 1

    data['parking'].fillna('нет_парковки', inplace=True)
    data['underground_parking'] = data['parking'].apply(lambda x: 1 if 'подземн' in x else 0)
    data['multilevel_parking'] = data['parking'].apply(lambda x: 1 if 'многоуровневая' in x else 0)
    data['barrier_parking'] = data['parking'].apply(lambda x: 1 if 'шлагбаум' in x else 0)

    data['offer_date'] = pd.to_datetime(data['offer_date'])
    data['year_week'] = data['offer_date'].apply(
        lambda x: datetime.date.fromisocalendar(*x.isocalendar()[:2], 1)
    )

    after_shape = data.shape

    if show_plots:
        g = data.groupby(['year_week', 'offer_type']).agg(
            {'offer_id': 'count', 'm2_price': ['mean', 'median']}
        ).reset_index()
        g.columns = ['year_week', 'offer_type', 'offer_amount', 'mean_m2_price', 'median_m2_price']
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(14, 5)

        for a in ax:
            plt.sca(a)
            plt.xticks(rotation=30)

        sns.lineplot(ax=ax[0], x=g['year_week'], y=g['mean_m2_price'], hue=g['offer_type'])
        ax[0].set_title('Dynamic of Mean prices of m2')
        sns.lineplot(ax=ax[1], x=g['year_week'], y=g['median_m2_price'], hue=g['offer_type'])
        ax[1].set_title('Dynamic of Median prices of m2')
        sns.lineplot(ax=ax[2], x=g['year_week'], y=g['offer_amount'], hue=g['offer_type'])
        ax[2].set_title('Dynamic of offer amount')
        plt.show()

    print('------------------------ Data info ------------------------')
    print('Pre shape:', pre_shape, 'After shape', after_shape)

    data = data[['offer_id', 'm2_price', 'city', 'n_square', 'remote_m',
                 'n_room_num', 'n_kitchen_area', 'n_living_area', 'n_floor',
                 'total_floor', 'n_built_in', 'n_ceiling', 'offer_type', 'house_type',
                 'renovation', 'balcony', 'has_concierge', 'apartments',
                 'underground_parking', 'multilevel_parking', 'barrier_parking', 'model_participation']]
    data = data.reset_index(drop=True)

    return data


def pre_train_formatting(df) -> ('DataFrame', list):
    """
    Функция для формирования фичей в предодготовленном дф
    -----------------------------------------------------
    df - дф, с коррекно для обучения заполнеными полями
    -----------------------------------------------------
    Возвращает:
    df - дф с сформированными фичами, поготовленный для отправки в модель для обучения
    segmentation_features - список фичей, не нужных для обучения, но нужных для категоризации
    """

    segmentation_features = ['offer_type', 'model_participation']

    target = ['m2_price']

    numeric_features = ['n_square', 'remote_m', 'n_room_num', 'n_kitchen_area', 'n_living_area',
                        'n_floor', 'total_floor', 'n_built_in', 'n_ceiling', 'has_concierge',
                        'apartments', 'underground_parking', 'multilevel_parking', 'barrier_parking']

    categorical_features = ['city', 'house_type', 'renovation', 'balcony']

    d = df[categorical_features]
    onehotencoder = OneHotEncoder()
    ohe_data = onehotencoder.fit_transform(d.values)
    ohe_data = pd.DataFrame(ohe_data.toarray(), columns=np.concatenate([*onehotencoder.categories_]))

    df = df[['offer_id'] + target + segmentation_features + numeric_features]
    df = pd.concat([df, ohe_data], axis=1)

    return df, segmentation_features


def evaluating(df: 'DataFrame', segmentation_features: list, show_plots=False) -> 'DataFrame':
    """
    Функция поиска недооцененных объектов
    -------------------------------------
    df - дф с сформированными фичами, поготовленный для отправки в модель для обучения
    offer_types - список, содержащий типы недвижимости (первичка/вторичка)
    segmentation_features - список, содержащий параметры, которые не участвуют в тренировке модели,
                            но нужны для формирования нужных сегментов из дф
    Возвращает:
    df2send - дф, содержащий записи с недооценненными объектами
    """

    def display_plot(df: 'DataFrame', t: str, param: str) -> None:
        """
        Функция для отображения графиков для оценки модели.
        --------------------------------------------------
        df - дф, на котором будут нарисованы графики
        t - тип недвижимости (первичка/вторичка)
        param - параметр даннаыых для обучения (train/test)
        """
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(14, 8)
        sns.histplot(ax=ax[0][0], x=df['rel_residuals'], kde=True)
        ax[0][0].set_title('Model residuals distribution', fontweight='bold')
        sns.regplot(ax=ax[0][1], data=df, x='m2_price', y='predicted_m2_price')
        ax[0][1].set_title('Assumption of Linear relationship between actual and predicted data', fontweight='bold')
        sns.scatterplot(ax=ax[1][0], x=df['m2_price'], y=df['rel_residuals'])
        ax[1][0].set_title('Residuals distribution', fontweight='bold')
        fig.suptitle(f'Model {t} {param} results visualization', fontweight='bold')
        plt.show()

    def creation_df_for_model(df: 'DataFrame', t: str, param: str) -> 'DataFrame':
        """
        Функция для формирования дф, нужного для модели в параметре param
        -----------------------------------------------------------------
        df - исходный дф с предобработанными данными до заливки их в модель
        t - тип недвижимости (первичка/вторичка)
        param - параметр данных для обучения (train/test)
        -----------------------------------------------------------------
        Возвращает дф, подготовленный для отправки в модель на обучение или прогноз в зависимости от param

        """
        tmp = df[(df['offer_type'] == t) & (df['model_participation'] == param)]
        offer_ids = tmp[['offer_id']]
        tmp.drop(columns=segmentation_features + ['offer_id'], inplace=True)

        return tmp, offer_ids

    def model_evaluating(t: str, y: 'Series', y_predicted: 'array', x: 'DataFrame') -> None:
        """
        Функция для вывода на эран показателей качетсва модели
        ------------------------------------------------------
        t - тип недвижимости (первичка/вторичка)
        y - фактическией значения оцениваемого параметра (m2_price)
        y_predicted - предсказанные значения оцениваемого параметра (m2_price)
        x - дф с фичами, по которым предсказываем оцениваемый параметр
        """
        print()
        print(f'------------------------ {t} model train score info ------------------------')
        print('Mean Absolute Percentage Error:', metrics.mean_absolute_percentage_error(y, y_predicted))
        print('Explained Variance Score, R2:', reg.score(x, y))

    def df_with_residuals_creation(tmp: 'DataFrame', offer_ids: 'DataFrame', y_predicted: 'array') -> 'DataFrame':
        """
        Функция для формирования дф с прогнозом и остатками
        tmp - дф, одним из полей которого должено быть m2_price
        offer_ids - дф, содержащий offer_id. Строки соответсвуют строкам дф tmp
        y_predicted - массив с предсказанными значениями признака m2_price
        ------------------------------------------------------
        Возвращает: обработанный дф с полями 'offer_id', 'm2_price', 'predicted_m2_price', 'abs_residuals', 'rel_residuals'
        """
        tmp = tmp[['m2_price']]
        tmp = pd.concat([offer_ids, tmp], axis=1)
        tmp['predicted_m2_price'] = y_predicted

        tmp['abs_residuals'] = tmp['m2_price'] - tmp['predicted_m2_price']
        tmp['rel_residuals'] = round((tmp['m2_price'] / tmp['predicted_m2_price']) - 1, 4)

        return tmp

    def zero_equality_residuals_check(tmp: 'DataFrame') -> None:
        """
        Функция проверки равенства нулю средного значения остатков модели
        -----------------------------------------------------------------
        tmp - дф, содержащий колонку остатков 'abs_residuals', среднее коротых мы будем оценивать
        -----------------------------------------------------------------
        Возвращает:
            - True, если нулевая гипотеза верна и среднее остатков статзначимо не отличается от 0
            - False, если нулевую гипотезу можно отвергнуть, т.е. среднее остатков не равно 0
        """
        alpha = 0.05
        m_control = 0
        m_treatment = tmp['abs_residuals'].mean()
        sigma_treatment = np.std(tmp['abs_residuals']) ** 2
        n = len(tmp['abs_residuals'])

        z_score = (m_treatment - m_control) / np.sqrt(sigma_treatment / n)

        # в зависимости от того, в какую строну смотреть (от знака z_score)
        # оцениваем вероятность получить такое или более выраженное значение
        if z_score >= 0:
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = stats.norm.cdf(z_score)

        result = False

        if alpha / 2 <= p_value:
            result = True

        print()
        print(f'------------------------ {t} residuals mean equals 0 check ------------------------')
        print(f'test result: {result}, p_value: {p_value}')

    # создаем дф, куда будем передавать записи с недооцененными объектами
    df2send = pd.DataFrame(columns=['offer_id', 'm2_price', 'predicted_m2_price', 'abs_residuals', 'rel_residuals'])

    # бежим в цикле по типам недвижимости
    for t in df['offer_type'].unique():

        # создание дф для тренировки
        param = 'train'
        tmp, offer_ids = creation_df_for_model(df, t, param)

        # тренировка модели
        y = tmp['m2_price']
        x = tmp.iloc[:, 1:]
        reg = LinearRegression().fit(x, y)
        y_predicted = reg.predict(x)

        # оценка качесва модели на тренировочных данных
        model_evaluating(t, y, y_predicted, x)

        # формирование итогововго дф с прогнозом цены
        tmp = df_with_residuals_creation(tmp, offer_ids, y_predicted)

        zero_equality_residuals_check(tmp)

        if show_plots:
            display_plot(tmp, t, param)

        # создание дф для прогноза
        param = 'test'
        tmp, offer_ids = creation_df_for_model(df, t, param)

        # прогноз значений на тестовых данных
        y = tmp['m2_price']
        x = tmp.iloc[:, 1:]
        y_predicted = reg.predict(x)

        # оценка прогноза тестовых данных
        model_evaluating(t, y, y_predicted, x)

        # формирование итогововго дф с прогнозом цены
        tmp = df_with_residuals_creation(tmp, offer_ids, y_predicted)

        if show_plots:
            display_plot(tmp, t, param)

        # оставляем объявления с дисконтом 10% или более по оценке модели
        tmp = tmp[tmp['rel_residuals'] <= -0.1]

        # передаем записи с недооцененными объектами в итоговый дф
        df2send = pd.concat([df2send, tmp])

    return df2send


def telegram_send_message(raw_data: 'DataFrame', df2send: 'DataFrame') -> None:
    """
    Функция для создания и отправки сообщения в тг
    raw_data - дф с необходимыми полями, в которых содержится человекочитаемая информация об объекте
    df2send - дф, который содержит объекты к отправке
    """

    data = df2send.merge(raw_data[['offer_id', 'title', 'city', 'url_offer', 'price', 'offer_type']], on='offer_id')
    data = data[
        ['title', 'city', 'url_offer', 'price', 'm2_price',
         'predicted_m2_price', 'rel_residuals', 'offer_type']
    ]

    # инициализация счетчика отправленных сообщений
    message_counter = 0

    # цикл, который бежит по объявлениям и отправляет сообщения в тг в нужном формате
    for t, c, uo, p, sp, psp, rr, ot in data.values:
        text = f"""
        <a href='{uo}'>{t}</a>
Price: {float(p) / 1000000} kk₽
m2 price: {round(float(sp) / 1000, 3)} k₽
Predicted m2 price: {round(float(psp) / 1000, 3)}k₽
Percentage error: {round(float(rr) * 100, 3)}%
Метро: "{c}"
Тип предложения: "{ot.upper()}"
        """
        # отправка сообщения в тг
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        requests.post(url=url, data={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'})

        # инкремент счетчика сообщений
        message_counter += 1

    # вывод на жкран количества отправленных сообщений
    print(message_counter, 'Messages have been sent')


def update_offers_status() -> None:
    """Функция обновления статуса объявления с test на train"""

    with sqlite3.connect(r'C:\Users\dmaryanovskiy\PycharmProjects\parsing\data_base\realty.db') as connection:
        cursor = connection.cursor()
        cursor.execute("update offers set model_participation = 'train'")
        connection.commit()

    data = pd.read_csv(r'C:\Users\dmaryanovskiy\PycharmProjects\parsing\data_base\realty_offers.csv')
    data['model_participation'] = 'train'
    data.to_csv(r'C:\Users\dmaryanovskiy\PycharmProjects\parsing\data_base\realty_offers.csv', index=False)


def realty_evaluation_bot(show_plots: bool = True) -> None:
    """Функция оценки стоимости квадртного метра объекта методом линейной регрессии"""

    # получаем сырые данные из БД
    raw_data = get_data()

    # преобразование сырых данных: исключение выбросов, заполнение пропусков и т.д.
    act_data = data_formatting(raw_data, show_plots)

    # проверка, есть ли данные для оценки
    if act_data[act_data['model_participation'] == 'test'].shape[0] == 0:
        print()
        print('------------------------ Нет объектов для оценки ------------------------')
        return

    # формирование фичей для обучения модели
    formatted_data, segmentation_features = pre_train_formatting(act_data)

    # поиск объектов с дисконтом при помощи линейной регрессии
    df2send = evaluating(formatted_data, segmentation_features, show_plots)

    # отправка сообщений в тг
    telegram_send_message(raw_data, df2send)

    # обновление статуса участия в модели
    update_offers_status()


if __name__ == '__main__':
    realty_evaluation_bot(show_plots=True)
