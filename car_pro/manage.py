import _sqlite3
import sqlite3

conn = sqlite3.connect("user_data.db", check_same_thread=False)
c = conn.cursor()
def create_usertabel():
    c.execute('CREATE TABLE IF NOT EXISTS usertable (username text ,password text)')

def add_userdata(username,password):
	c.execute('INSERT INTO usertable(username,password) VALUES (?,?)',(username,password))
	conn.commit()



def login_user(username,password):
	c.execute('SELECT * FROM usertable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data



