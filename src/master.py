from src.data_management.ihub_data import IhubData
from src.data_management.stock_data import StockData
from src.data_management.combine_data import CombineData


if __name__ == '__main__':
    stock_lst = ['cbyi','mine','xtrn','dolv','pgpm','cnxs','amlh','exol','coho',
        'uoip','kget']

    cbyi = 'Cal-Bay-International-Inc-CBYI-5520'
    mine = 'Minerco-Inc-MINE-17939'
    xtrn = 'Las-Vegas-Railway-Express-XTRN-16650'
    dolv = 'Dolat-Ventures-Inc-DOLV-16401'
    pgpm = 'Pilgrim-Petroleum-Corp-PGPM-5655'
    cnxs = 'Connexus-Corp-CNXS-17863'
    amlh = 'American-Leisure-Holdings-Inc-AMLH-29447'
    exol = 'EXOlifestyle-Inc-EXOL-11015'
    coho = 'Crednology-Holding-Corp-COHO-4899'
    uoip = 'UnifiedOnline-Inc-UOIP-5196'
    kget = 'CaliPharms-Inc-KGET-10313'

    ihub_lst = [cbyi,mine,xtrn,dolv,pgpm,cnxs,amlh,exol,coho,uoip,kget]

    # for stock in ihub_lst:
    #     data = IhubData(stock,verbose = 1)
    #     data.pull_posts()

    print ('getting pricing data...')

    for stock in stock_lst:
        s = StockData(stock)
        s.add_stock_data()
        combined_data = CombineData(stock)
        combined_data.compile_data()
