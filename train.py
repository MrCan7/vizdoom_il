from models.basemodel import BaseModel
from data.data_prep import get_train_val_test
from data.data_loader import VizDoomData

if __name__ == "__main__":
    model = BaseModel()
    df_train, df_val, df_test = get_train_val_test()

    train_data = VizDoomData(df_train)
    val_data = VizDoomData(df_val, istest= True)
    test_data = VizDoomData(df_test, istest= True)
    print(train_data)