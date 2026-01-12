import streamlit as st
from sqlalchemy import create_engine, text

def main():
    cfg = st.secrets["mysql"]

    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{int(cfg.get('port', 3306))}/{cfg['database']}?charset=utf8mb4"
    )

    engine = create_engine(url, pool_pre_ping=True)

    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).fetchone()
        print("✅ Conectado a MySQL")
        print("MySQL VERSION:", version[0])

        tables = conn.execute(text("SHOW TABLES")).fetchmany(10)
        print("Tablas (muestra):")
        for t in tables:
            print(" -", t[0])

if __name__ == "__main__":
    main()
