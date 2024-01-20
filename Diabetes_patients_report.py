from ydata_profiling import ProfileReport
import pandas as pd df = pd.read_csv("dataset/diabetes2.csv")
profile = ProfileReport(df, title = "Diabetes patients report")
profile.to_file("Diabetes _patients_report.html")

