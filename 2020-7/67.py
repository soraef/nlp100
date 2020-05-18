import csv 
import gensim
from sklearn.cluster import KMeans

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

countries = []
country_vectors = []

with open("country.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        country = row[0].split(",")[0].split("(")[0].replace(" ", "_")
        try:
            # print(model[country])
            country_vectors.append(model[country])
            countries.append(country)
        except KeyError:
            print(f"{country} not in vocabukary")

pred = KMeans(n_clusters=5).fit_predict(country_vectors)

cluster = [ data for data in zip(countries, pred)]

sorted_cluster = sorted(cluster, key=lambda x: x[1])

cluster_num = -1
for item in sorted_cluster:
    if cluster_num != item[1]:
        cluster_num += 1
        print(f"== CLUSTER {cluster_num} ==")
    print(item[0])

            
# Antigua_and_Barbuda not in vocabukary
# Bosnia_and_Herzegovina not in vocabukary
# Central_African_Republic not in vocabukary
# Côte_d'Ivoire not in vocabukary
# Guinea-Bissau not in vocabukary
# Holy_See_ not in vocabukary
# Lao_People's_Democratic_Republic not in vocabukary
# Northern_Cyprus not in vocabukary
# Palestinian_Territory not in vocabukary
# Papua_New_Guinea not in vocabukary
# Russian_Federation not in vocabukary
# Saint_Kitts_and_Nevis not in vocabukary
# Saint_Vincent_and_the_Grenadines not in vocabukary
# Sao_Tome_and_Principe not in vocabukary
# South_Sudan not in vocabukary
# Syrian_Arab_Republic not in vocabukary
# Timor-Leste not in vocabukary
# Trinidad_and_Tobago not in vocabukary
# == CLUSTER 0 ==
# Argentina
# Bahamas
# Barbados
# Belize
# Bolivia
# Brazil
# Canada
# Cape_Verde
# Chile
# Colombia
# Costa_Rica
# Cuba
# Dominica
# Dominican_Republic
# Ecuador
# El_Salvador
# Grenada
# Guatemala
# Guyana
# Haiti
# Honduras
# Jamaica
# Mexico
# Nicaragua
# Panama
# Paraguay
# Peru
# Saint_Lucia
# Suriname
# Uruguay
# Venezuela
# == CLUSTER 1 ==
# Abkhazia
# Albania
# Andorra
# Armenia
# Austria
# Azerbaijan
# Belarus
# Belgium
# Bulgaria
# Croatia
# Cyprus
# Czechia
# Denmark
# Estonia
# Finland
# France
# Georgia
# Germany
# Greece
# Hungary
# Iceland
# Ireland
# Italy
# Kazakhstan
# Kosovo
# Latvia
# Liechtenstein
# Lithuania
# Luxembourg
# Macedonia
# Malta
# Moldova
# Monaco
# Montenegro
# Nagorno_Karabagh_Republic
# Netherlands
# Norway
# Poland
# Portugal
# Pridnestrovian_Moldavian_Republic
# Romania
# San_Marino
# Serbia
# Slovakia
# Slovenia
# South_Ossetia
# Spain
# Sweden
# Switzerland
# Turkey
# Ukraine
# == CLUSTER 2 ==
# Algeria
# Angola
# Benin
# Botswana
# Burkina_Faso
# Burundi
# Cameroon
# Comoros
# Congo
# Congo
# Djibouti
# Equatorial_Guinea
# Eritrea
# Ethiopia
# Gabon
# Gambia
# Ghana
# Guinea
# Kenya
# Lesotho
# Liberia
# Madagascar
# Malawi
# Mali
# Mauritania
# Mauritius
# Mozambique
# Namibia
# Niger
# Nigeria
# Rwanda
# Senegal
# Seychelles
# Sierra_Leone
# Somalia
# Somaliland
# South_Africa
# Sudan
# Swaziland
# Tanzania
# Togo
# Tunisia
# Uganda
# Western_Sahara
# Zambia
# Zimbabwe
# == CLUSTER 3 ==
# Afghanistan
# Australia
# Bahrain
# Bangladesh
# Bhutan
# Brunei_Darussalam
# Cambodia
# Chad
# China
# Egypt
# India
# Indonesia
# Iran
# Iraq
# Israel
# Japan
# Jordan
# Korea
# Korea
# Kuwait
# Kyrgyzstan
# Lebanon
# Libya
# Malaysia
# Maldives
# Mongolia
# Morocco
# Myanmar
# Nepal
# Oman
# Pakistan
# Philippines
# Qatar
# Saudi_Arabia
# Singapore
# Sri_Lanka
# Taiwan
# Tajikistan
# Thailand
# Turkmenistan
# United_Arab_Emirates
# United_Kingdom
# United_States
# Uzbekistan
# Viet_Nam
# Yemen
# == CLUSTER 4 ==
# Cook_Islands
# Fiji
# Kiribati
# Marshall_Islands
# Micronesia
# Nauru
# New_Zealand
# Niue
# Palau
# Samoa
# Solomon_Islands
# Tonga
# Tuvalu
# Vanuatu