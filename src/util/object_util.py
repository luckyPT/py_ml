import pickle
import os


class ObjectUtil:
    @staticmethod
    def save_obj(obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_obj(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    file_name = "./dict.pickle"
    d = {"a": 1, "b": 2, "c": 3}
    ObjectUtil.save_obj(d, file_name)
    del d
    d = ObjectUtil.load_obj(file_name)
    print(d)
    os.remove(file_name)
