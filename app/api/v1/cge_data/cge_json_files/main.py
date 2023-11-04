from bs4 import BeautifulSoup, Tag
import requests
import json
from datetime import date, timedelta
import os
import time


def get_jsons_by_day(dia, mes, ano):
    url_data = "https://www.cgesp.org/v3/alagamentos.jsp?dataBusca={}%2F{}%2F{}&enviaBusca=Buscar".format(dia, mes, ano)
    response = requests.get(url_data)
    lista_pontos = []

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')

        div_alagamentos = soup.find('div', {'class': 'col-alagamentos'})

        if div_alagamentos:

            tabelas = div_alagamentos.find_all('table', {'class': 'tb-pontos-de-alagamentos'})

            for tabela in tabelas:
                ponto = {}
                bairro_info = tabela.find('td', {'class': 'bairro arial-bairros-alag linha-pontilhada'})
                total_pts_info = tabela.find('td', {'class': 'total-pts arial-bairros-alag'})
                ponto_de_alagamento_info = tabela.find('div', {'class': 'ponto-de-alagamento'})

                if bairro_info:
                    ponto['Bairro'] = bairro_info.text.strip()
                if total_pts_info:
                    ponto['Total Pts'] = total_pts_info.text.strip()
                if ponto_de_alagamento_info:
                    ul_info = ponto_de_alagamento_info.find('ul')
                    if ul_info:
                        lis = ul_info.find_all('li')
                        if lis:
                            ponto['Status'] = lis[0].get('title', 'Desconhecido')
                            horario_rua = []
                            for index, content in enumerate(lis[2].contents):
                                if isinstance(content, Tag):  
                                    continue
                                horario_rua.append(content.strip()) 

                            ponto['Horario'] = horario_rua[0]
                            ponto['Rua'] = horario_rua[1]
                            ponto['Data'] = "{}/{}/{}".format(dia, mes, ano)    
                            ponto['Sentido'] = lis[4].text.strip().split('Referência:')[0].strip().split(":")[1]
                            ponto['Referencia'] = lis[4].text.strip().split('Referência:')[1].strip() if 'Referência:' in lis[4].text else 'Desconhecido'

                lista_pontos.append(ponto)


            json_str = lista_pontos
            return json_str

        else:
            print("Div com a classe 'col-alagamentos' não encontrado.")
    else:
        print("Falha ao fazer o request. Status code:", response.status_code)

    return None

def generate_json():
    start_date = date(2022, 5, 31)
    end_date = date.today()
    delta = timedelta(days=1)

    while start_date <= end_date:
       
        data = get_jsons_by_day(start_date.day, start_date.month, start_date.year)

        if data: 
            directory_path = os.path.join(str(start_date.year), f"{start_date.month:02}")
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            file_name = f"{start_date.year}_{start_date.month:02}_{start_date.day:02}.json"
            file_path = os.path.join(directory_path, file_name)
            print(file_path)
            with open(file_path, 'w') as file:
                json.dump(data, file)

    
        start_date += delta
        time.sleep(0.500)


if __name__ == '__main__':
    generate_json()