# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:32:35 2024

@author: samjo
"""

import pandas as pd 

dogFrame = pd.DataFrame({'Weight':{'Fido':14.3,
                                   'Rex':85.1,
                                   'Scooby':121.1},
                         'Age':{'Fido':10,
                                'Rex':8,
                                'Scooby':4},
                         'Breed':{'Fido':'Jack Russel',
                                  'Rex':'German Shepherd',
                                  'Scooby':'Great Dane'}})
print(dogFrame)