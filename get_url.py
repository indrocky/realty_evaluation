from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

import time


class GetUrl:
    def __init__(self, driver, realty_type, year='2000'):
        self.driver = driver
        self.realty_type = realty_type
        self.year = year
        self.url = None

    def checkbox_button_click(self, params):
        while True:
            self.driver.find_element(By.XPATH, f"//span[@data-marker='{params}/text']").click()
            if self.driver.find_element(By.XPATH, f"//input[@data-marker='{params}/input']").is_selected():
                break

    def realty_type_click(self):
        print('Клик по кнопке выбора типа недвижимости')
        WebDriverWait(self.driver, 10).until(
            ec.presence_of_element_located((By.XPATH, "//input[@data-marker='params[499]']//ancestor::label"))
        ).click()
        time.sleep(1)

        if self.realty_type == 'вторичка':
            print('Клик по кнопке "Вторичка"')
            self.driver.find_element(By.XPATH, "//div/span[text()='Вторичка']").click()

        elif self.realty_type == 'первичка':
            self.driver.find_element(By.XPATH, "//div/span[text()='Новостройка']").click()

        else:
            print('Unknown realty type')
            self.driver.close()

        time.sleep(1)


    def filter_button(self):
        print('Клик по кнопке с фильтрами')
        WebDriverWait(self.driver, 10).until(
            ec.presence_of_element_located((By.XPATH, "//div/h1[contains(text(), 'Недвижимость')]//following::button"))
        ).click()
        time.sleep(1)

    # def realty_type_click(self):
    #     if self.realty_type == 'вторичка':
    #         print('клик по кнопке "Вторичка"')
    #         WebDriverWait(self.driver, 10).until(
    #             ec.presence_of_element_located((By.XPATH, "//span[@data-marker='params[499](5254-radio)/text']"))
    #         ).click()
    #         time.sleep(1)
    #
    #     elif self.realty_type == 'первичка':
    #         print('клик по кнопке "Новостройки"')
    #         WebDriverWait(self.driver, 10).until(
    #             ec.presence_of_element_located((By.XPATH, "//span[@data-marker='params[499](5255-radio)/text']"))
    #         ).click()
    #         time.sleep(1)
    #
    #     else:
    #         print('Unknown realty type')
    #         self.driver.close()

    def rooms_checkbox(self):
        print('клик по чекбоксу "2 комнаты"')
        self.checkbox_button_click('params[549](5697)')
        time.sleep(1)
        print('клик по чекбоксу "3 комнаты"')
        self.checkbox_button_click('params[549](5698)')
        time.sleep(1)

    def foot_walking(self):
        # Скрол для отображения на экране удаленности от метро
        element = self.driver.find_element(By.XPATH, "//span[@data-marker='params[110688]()/text']")
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

        print('клик по кнопке "Пешком до метро мин. = 20"')
        self.checkbox_button_click('footWalkingMetro(20)')
        time.sleep(1)

    def ceiling_height(self):
        # Скрол для отображения на экране Высоты потолков
        element = self.driver.find_element(By.XPATH, "//span[@data-marker='footWalkingMetro(20)/text']")
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

        print('клик по кнопке "Высота потолков м = 2,7"')
        self.checkbox_button_click('params[122375](2911292)')
        time.sleep(1)

    def house_materials(self):
        # Скрол для отображения типа дома
        element = self.driver.find_element(By.XPATH, "//span[@data-marker='params[110687](458588)/text']")
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

        print('Клик по кнопке "Тип дома = кирпичный"')
        self.checkbox_button_click('params[498](5244)')
        time.sleep(1)
        print('Клик по кнопке "Тип дома = монолитный"')
        self.checkbox_button_click('params[498](5247)')
        time.sleep(1)
        print('Клик по кнопке "Тип дома = кирпично-монолитный"')
        self.checkbox_button_click('params[498](2308811)')
        time.sleep(1)

    def year_built(self):
        # Скрол для отображения Года постройки дома
        element = self.driver.find_element(By.XPATH, "//span[@data-marker='params[122375](2911292)/text']")
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)

        print('Ввод текста в поле "Год постройки дома"', self.year)
        element = self.driver.find_element(By.XPATH, "//input[@data-marker='params[110499]/from']")
        element.send_keys(Keys.CONTROL, 'a')
        element.send_keys(self.year)
        time.sleep(1)

    def submit_click(self):
        print('Клик по кнопке "Показать"')
        time.sleep(3)
        elements = self.driver.find_elements(By.XPATH, "//button[@type='button']")
        for e in elements:
            if 'Показать' in e.text:
                e.click()
                break
        time.sleep(1)

    def show_url(self):
        self.url = self.driver.current_url
        print(f'Полученный url {self.url}')
        return self.url

    def close(self):
        print('webdriver session closed')
        self.driver.close()


def get_url(realty_type: str):

    if realty_type not in ['первичка', 'вторичка']:
        print('Введите категорию объявлений: "первичка" или "вторичка"')
        return None
    else:
        exe_path = r'chromedriver.exe'
        driver = webdriver.Chrome(executable_path=exe_path)
        driver.get('https://www.avito.ru/sankt-peterburg/nedvizhimost')
        time.sleep(3)

        if realty_type == 'вторичка':
            url_session = GetUrl(driver, realty_type)
            url_session.realty_type_click()
            url_session.filter_button()
            # url_session.rooms_checkbox()
            # url_session.foot_walking()
            # url_session.ceiling_height()
            url_session.year_built()
            url_session.house_materials()
            url_session.submit_click()
            url_session.show_url()
            url = url_session.url
            url_session.close()

        elif realty_type == 'первичка':
            url_session = GetUrl(driver, realty_type)
            url_session.realty_type_click()
            url_session.filter_button()
            # url_session.rooms_checkbox()
            # url_session.foot_walking()
            # url_session.ceiling_height()
            url_session.house_materials()
            url_session.submit_click()
            url_session.show_url()
            url = url_session.url
            url_session.close()

        else:
            print('Oops, something wrong here')
            return None

    return url
