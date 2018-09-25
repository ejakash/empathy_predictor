known_string_types = {'Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet usage', 'Gender', 'Left - right handed',
                      'Education', 'Only child', 'Village - town', 'House - block of flats'}
Y_index = "Empathy"
input_file = 'input/responses.csv'

# Non binary values are sorted in some order.
text_values = {'Gender': ['female', 'male'], 'Only child': ['no', 'yes'], 'Village - town': ['village', 'city'],
               'House - block of flats': ['block of flats', 'house/bungalow'],
               'Left - right handed': ['right handed', 'left handed'],
               'Smoking': ['never smoked', 'tried smoking', 'former smoker', 'current smoker'],
               'Alcohol': ['never', 'social drinker', 'drink a lot'],
               'Punctuality': ['i am often running late', 'i am often early', 'i am always on time'],
               'Lying': ['never', 'sometimes', 'only to avoid hurting someone', 'everytime it suits me'],
               'Internet usage': ['no time at all', 'less than an hour a day', 'few hours a day', 'most of the day'],
               'Education': ['currently a primary school pupil', 'primary school', 'secondary school',
                             'college/bachelor degree', 'masters degree', 'doctorate degree']}

