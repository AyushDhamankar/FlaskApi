# from flask import Flask,request
# from flask_restful import Resource, Api
# import pickle
# import pandas as pd
# from flask_cors import CORS
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import numpy as np

# app = Flask(__name__)
# #
# CORS(app)
# # creating an API object
# api = Api(app)

# #prediction api call
# # class prediction(Resource):
# #     def get(self, budget):
# #         # Load the trained model
# #         model = pickle.load(open('simple_linear_regression.pkl', 'rb'))

# #         # Ensure that the feature name matches the one used during training
# #         budget_feature_name = 'Marketing Budget (X) in Thousands'

# #         # Create a DataFrame with the correct feature name
# #         budget_df = pd.DataFrame({'Marketing Budget (X) in Thousands': [budget]})

# #         # Make the prediction
# #         prediction = model.predict(budget_df)
# #         prediction = int(prediction[0])
# #         print(prediction)
# #         return str(prediction)

# #
# # class prediction(Resource):
# #     def get(self, budget):
# #         # Load the trained model
# #         model = pickle.load(open('bhp.pickle', 'rb'))

# #         # Create a DataFrame with the correct feature name
# #         budget_df = pd.DataFrame({'Marketing Budget (X) in Thousands': [budget]})

# #         # Make the prediction
# #         prediction = model.predict(budget_df)
# #         prediction = int(prediction[0])
# #         print(prediction)
# #         return str(prediction)

# #try
# class prediction(Resource):
#     def get(self, location, sqft, bhk, bath):
#         # Load the trained model
#         df10 = pd.read_csv('bhp-final.csv')
#         dummies = pd.read_csv('bhp-dummies.csv')

#         df11 = pd.concat([df10,dummies.drop('Other',axis='columns')],axis='columns')
#         df12 = df11.drop('location',axis='columns')

#         X = df12.drop(['price'],axis='columns')
#         y = df12.price

#         X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
#         lr_clf = LinearRegression()
#         lr_clf.fit(X_train,y_train)
#         lr_clf.score(X_test,y_test)

#         loc_index = np.where(X.columns==location)[0][0]

#         x = np.zeros(len(X.columns))
#         x[0] = sqft
#         x[1] = bath
#         x[2] = bhk
#         if loc_index >= 0:
#             x[loc_index] = 1

#         return lr_clf.predict([x])[0]

#         # model = pickle.load(open('bhp.pickle', 'rb'))

#         # # Create a DataFrame with the correct feature name
#         # budget_df = pd.DataFrame({'Marketing Budget (X) in Thousands': [budget]})

#         # # Make the prediction
#         # prediction = model.predict(budget_df)
#         # prediction = int(prediction[0])
#         # print(prediction)
#         # return str(prediction)

# #data api
# class getData(Resource):
#     def get(self):
#             df = pd.read_csv('bhp-final.csv')
#             #df =  df.rename({'Marketing Budget': 'budget', 'Actual Sales': 'sale'}, axis=1)  # rename columns
#             #print(df.head())
#             #out = {'key':str}
#             res = df.to_json(orient='records')
#             #print( res)
#             return res

# #
# api.add_resource(getData, '/api')
# api.add_resource(prediction, '/prediction/<string:l>/<int:a>/<int:bh>/<int:ba>')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask
from flask_restful import Resource, Api, reqparse
import pickle
import pandas as pd
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

CORS(app)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data', type=str)  # Define the expected data field

class prediction(Resource):
    def get(self, location, sqft, bhk, bath):
        df10 = pd.read_csv('bhp-final.csv')
        dummies = pd.read_csv('bhp-dummies.csv')

        df11 = pd.concat([df10, dummies.drop('Other', axis='columns')], axis='columns')
        df12 = df11.drop('location', axis='columns')

        X = df12.drop(['price'], axis='columns')
        y = df12.price

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        lr_clf = LinearRegression()
        lr_clf.fit(X_train, y_train)
        lr_clf.score(X_test, y_test)

        # Check if the location column exists
        if location in X.columns:
            loc_index = np.where(X.columns == location)[0][0]
            x = np.zeros(len(X.columns))
            x[0] = sqft
            x[1] = bath
            x[2] = bhk
            if loc_index >= 0:
                x[loc_index] = 1
                return str(lr_clf.predict([x])[0])
        else:
            return "Location not found in the dataset"

class getData(Resource):
    def get(self):
        df = pd.read_csv('bhp-final.csv')
        res = df.to_json(orient='records')
        return res
    
class sendData(Resource):
    def send(self):
        args = parser.parse_args()
        received_data = args['data']
        # Process the received data
        return {'message': 'Data received successfully', 'data': received_data}

api.add_resource(getData, '/api')
api.add_resource(prediction, '/prediction/<string:location>/<int:sqft>/<int:bhk>/<int:bath>')
api.add_resource(sendData, '/receive_data')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

