from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and mappings
model = pickle.load(open('laptop_price_model.pkl', 'rb'))

# Mappings used during training
cpu_map = {'Intel Core i7': 0, 'Intel Core i5': 1, 'Intel Core i3': 2, 'Other Intel Processor': 3, 'AMD Processor': 4}
gpu_map = {'Nvidia': 0, 'Intel': 1, 'AMD': 2}
os_map = {'Windows': 0, 'Mac': 1, 'Linux': 2, 'No OS': 3}
company_map = {'Lenovo': 0, 'HP': 1, 'Dell': 2, 'Asus': 3, 'Acer': 4, 'MSI': 5, 'Apple': 6, 'Other': 7}
typename_map = {'Ultrabook': 0, 'Notebook': 1, 'Gaming': 2, '2 in 1 Convertible': 3, 'Workstation': 4, 'Netbook': 5}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    companies = list(company_map.keys())
    typenames = list(typename_map.keys())
    cpus = list(cpu_map.keys())
    gpus = list(gpu_map.keys())
    oses = list(os_map.keys())
    rams = ['2', '4', '8', '16', '32', '64']
    storages = ['0', '128', '256', '512', '1024']

    if request.method == 'POST':
        form_data = request.form

        company = company_map[form_data['company']]
        typename = typename_map[form_data['typename']]
        inches = float(form_data['inches'])
        touchscreen = 1 if form_data['touchscreen'] == 'Yes' else 0
        ips = 1 if form_data['ips'] == 'Yes' else 0
        res = form_data['resolution']
        x_res, y_res = map(int, res.lower().split('x'))
        ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / inches

        cpu = cpu_map[form_data['cpu']]
        ram = int(form_data['ram'])
        gpu = gpu_map[form_data['gpu']]
        os = os_map[form_data['os']]
        weight = float(form_data['weight'])
        hdd = int(form_data['hdd'])
        ssd = int(form_data['ssd'])
        hybrid = 1 if form_data.get('hybrid') == 'on' else 0
        flash = 1 if form_data.get('flash') == 'on' else 0

        input_data = np.array([[company, typename, inches, touchscreen, ips, ppi,
                                cpu, ram, gpu, os, weight, hdd, ssd, hybrid, flash]])
        prediction = int(model.predict(input_data)[0])

    return render_template('index.html',
                           companies=companies,
                           typenames=typenames,
                           cpus=cpus,
                           gpus=gpus,
                           oses=oses,
                           rams=rams,
                           storages=storages,
    prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)