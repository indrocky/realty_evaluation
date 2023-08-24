# other
import time

# self_writing
from get_url import get_url
from get_json import get_json
from get_offers import get_offers
from data_save import get_df, data_save_csv, data_load_to_db


def parsing(*args):
    site = 'https://www.avito.ru'

    start = time.time()

    new = 'первичка'
    secondary = 'вторичка'
    diff_set = set(args).difference({new, secondary})

    if len(diff_set) != 0:
        print('wrong offer_types inputed')
        print(diff_set)
        end = time.time()
        print(f'script time {end - start} seconds')

        return

    offers_amount = 0

    for offer_type in args:
        cnt = 0
        url = None

        while cnt < 5:
            cnt += 1
            try:
                url = get_url(offer_type)
                print(f'{cnt} попытка получить URL страницы успешна')
                print(url)
                break
            except Exception as exc:
                print(f'{cnt} попытка получить URL страницы не удалась', type(exc), exc.args)

        if url is not None:
            i = 1
            last_page = 100 # хардкод последней страницы. Нужно усовершенствовать скрипт, чтобы последня страница находилась самостоятельно

            while i <= last_page:
                print('page number:', i)
                params = {'p': i, 's': 104}

                cnt = 0
                status = None

                while cnt < 5:
                    try:
                        session_data, status, last_page_attempt = get_json(url, params)

                        # if last_page_attempt is not None:
                        #     last_page = last_page_attempt

                        if session_data[0] is not None:
                            break

                    except Exception as exc:
                        print(f'{cnt} attempt get_json error', type(exc), exc.args)

                if status != 200:
                    print('parsing.py connection unsuccessful')
                    break

                cnt = 0
                offers_data = []

                while cnt < 5:
                    cnt += 1
                    try:
                        print(f'last page {offer_type}:', last_page)
                        offers_data = get_offers(*session_data, site, offer_type, new, secondary)
                        break
                    except Exception as exc:
                        print(f'{cnt} attempt get_offers error', type(exc), exc.args)

                if len(offers_data) == 0:
                    print("New offers haven't been added")
                    print()

                else:
                    df = get_df(offers_data)
                    offers_amount += data_save_csv(df)
                    data_load_to_db(df)
                    print()

                i += 1

                if i != last_page:
                    time.sleep(10)

    print(f'{offers_amount} offers have been added')

    end = time.time()
    print(f'script time {round((end - start) / 60, 2)} minutes')
    print()