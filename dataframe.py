import pandas as pd
from .general import Chronometer, printColumnsInCell

class Orangutans(pd.DataFrame):   
    """
    Avoid colomn names repetition in transformations
    Keep transformations tracked and documented 
    Load from saved parquet object or from database
    """
    def __init__(self, args_, report_status = True):
        pd.DataFrame.__init__(self, args_)
        self.report_status = report_status
        self.report_str = ''
    
    def transformer(self, col=None, transfo=lambda x: x, logic=None, 
                  col1=None, col2=None, 
                  logic1=None, logic2=None, univers='', category='', comments='', tests=None, alone=True, dependency=[]):
    
        if not col is None:
            col1 = col
            col2 = col

        if not logic is None:
            logic1 = logic
            logic2 = logic
        
        chrono = Chronometer()
        if alone :
            prompt = 'Transfo. {} ({}/{})'.format(col1, univers, category)
            print(prompt)
            if self.report_status:
                self.report_str += prompt+'\n'
            chrono.start(name='=> took ')
        else :
            chrono.start(name='§ {} took ')
            
        if not(logic2 is None or col2 is None):
            to_insert = transfo(self[logic2][col2])
        elif not(logic2 is None) and col2 is None:
            to_insert = transfo(self[logic2])
        elif logic2 is None and not(col2 is None):
            to_insert = transfo(self[col2])
        else:
            to_insert = transfo(self)

        if not tests is None:
            tests.run(to_insert)

        if not(logic1 is None or col1 is None):
            self.loc[logic1,col1] = to_insert
        elif not(logic1 is None) and col1 is None:
            self.loc[logic1,:] = to_insert
        elif logic1 is None and not(col1 is None):
            self.loc[:,col1] = to_insert
        else:
            self.loc[:] = to_insert
        prompt = chrono.display_str()
        print(prompt)
        self.report_str += prompt+'\n'
    
    def writeReport(self, path='/data/2_process/report'):
        with open(path+'.txt', 'w') as a:
            a.write(self.report_str)
        
    def multiTransformer(self, list_transfo):
        univers = list_transfo[0]['univers']
        category = list_transfo[0]['category']
        prompt = '/nTransfo. {} / {}'.format(univers, category)
        self.report_str += prompt+'\n'
        print(prompt)
        for t in list_transfo:
            if t['univers']==univers and t['category']==category:
                self.transformer(alone=False, **t)
            else: 
                prompt = '/nTransfo. {} / {}'.format(univers, category)
                print(prompt)
                self.report_str += prompt+'\n'
                self.transformer(alone=False, **t)
                univers = t['univers']
                category = t['category']
        prompt = '/n'
        print(prompt)
        self.report_str += prompt+'\n'
        
    def write(self, name):
        chrono = Chronometer()
        chrono.start(name='Writing {} took ')
        self.to_parquet('{}.parquet'.format(name))
        self.dtypes.to_csv('{}_dtypes.csv'.format(name), index_label=False)
        prompt = chrono.display_str()
        print(prompt)
        self.report_str += prompt+'\n'
        
    def read(name : str, col : list = None, keep_logic = None):
        chrono = Chronometer()
        chrono.start(name='Reading {} took ')
        types = pd.read_csv('{}_dtypes.csv'.format(name)).to_dict()['0']
        if not col is None:
            types = {t:types[t] for t in col}
        if not keep_logic is None :
            out = Orangutan(pd.read_parquet('{}.parquet'.format(name), columns=col).astype(
                    types)[keep_logic])
        else :
            out = Orangutan(pd.read_parquet('{}.parquet'.format(name), columns=col).astype(
                    types))
        chrono.display()
        return out

    def replaceColumnName(self, before, after):
        col = list(self.columns)
        col[col.index(before)]=after
        self.columns = col

    def autoMult(self, targeted, applied):
        self.loc[:, targeted] = self[targeted]*self[applied].values

    def autoDiv(self, targeted, applied):
        self.loc[:, targeted] = self[targeted]/self[applied].values

    def readSQL(sql: str, connector, name='DB'):
        chrono = Chronometer()
        chrono.start(name='{} request'.format(name))
        result = pd.read_sql(sql,connector)
        chrono.display()
        return Orangutan(result)

    def printColumns(self): return printColumnsInCell(list(self.columns))
    
    def filtering(self, col_del=[], keep_logic=None): 
        if not keep_logic is None:
            self.drop(columns=col_del, index=self.index[~keep_logic], inplace=True)
        else:
            self.drop(columns=col_del, inplace=True)