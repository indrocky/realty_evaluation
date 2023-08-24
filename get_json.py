# web
import requests
from selectolax.parser import HTMLParser
from hyper.contrib import HTTP20Adapter
from urllib.parse import unquote
from fake_headers import Headers

# other
import json
import datetime
import time


def get_json(url, params):
    """
    Функция для получения HTML кода страницы с объявлениями.
    Возвращает HTML-код, объект сессии, заголовок сессии, и статус соединения
    """

    print(datetime.datetime.now())
    header = Headers(headers=False).generate()
    session = requests.Session()
    session.mount('https://', HTTP20Adapter())

    i = 0
    status = None
    request = None

    while i < 5:
        time.sleep(5)
        i += 1
        try:
            request = session.get(url=url, headers=header, params=params)
            print(f'{i} attempt connection')
            status = request.status_code
            break
        except Exception as exc:
            print(f'{i} connection', exc.args)

    if status is None:
        return [None] * 3, None, None

    elif status != 200:
        print(request)
        return [None] * 3, [status], None

    else:
        print(request.url)
        request.encoding = 'utf-8'
        html = request.text
        tree = HTMLParser(html)
        scripts = tree.css('script')

        data = None
        last_page = None

        for script in scripts:
            if 'window.__initialData__' in script.text():
                jsontext = (script.text().split(';')[0].split('=')[-1].strip())
                jsontext = unquote(jsontext)
                jsontext = jsontext[1:-1]

                data = json.loads(jsontext)

        # if data is not None:
        #     page_nodes = tree.css('span')
        #
        #     for i in range(len(page_nodes)):
        #         if 'След.' in page_nodes[i].text():
        #             try:
        #                 last_page = int(page_nodes[i - 1].text())
        #             except:
        #                 last_page = None

        last_page = None
        return [data, session, header], status, last_page
