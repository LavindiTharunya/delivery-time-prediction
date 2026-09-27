"""
Interactive Delivery Time Prediction Web Application (Gradio)
Rebuilt using exclusively real columns from the Olist Brazilian E-Commerce dataset.
"""

import gradio as gr
from datetime import datetime, timedelta
import pandas as pd
from typing import Tuple

from src.predict import get_predictor
from src.features import BRAZILIAN_STATE_COORDS

# Load predictor instance
predictor = get_predictor()
categories = predictor.get_available_categories()
states = predictor.get_available_states()

# State names mapping for intuitive UI
STATE_NAMES = {
    'AC': 'Acre (AC)',
    'AL': 'Alagoas (AL)',
    'AM': 'Amazonas (AM)',
    'AP': 'Amapá (AP)',
    'BA': 'Bahia (BA)',
    'CE': 'Ceará (CE)',
    'DF': 'Distrito Federal (DF)',
    'ES': 'Espírito Santo (ES)',
    'GO': 'Goiás (GO)',
    'MA': 'Maranhão (MA)',
    'MG': 'Minas Gerais (MG)',
    'MS': 'Mato Grosso do Sul (MS)',
    'MT': 'Mato Grosso (MT)',
    'PA': 'Pará (PA)',
    'PB': 'Paraíba (PB)',
    'PE': 'Pernambuco (PE)',
    'PI': 'Piauí (PI)',
    'PR': 'Paraná (PR)',
    'RJ': 'Rio de Janeiro (RJ)',
    'RN': 'Rio Grande do Norte (RN)',
    'RO': 'Rondônia (RO)',
    'RR': 'Roraima (RR)',
    'RS': 'Rio Grande do Sul (RS)',
    'SC': 'Santa Catarina (SC)',
    'SE': 'Sergipe (SE)',
    'SP': 'São Paulo (SP)',
    'TO': 'Tocantins (TO)'
}

STATE_CHOICES = [(f"{code} - {STATE_NAMES.get(code, code)}", code) for code in states]


def format_prediction(
    customer_state_choice: str,
    seller_state_choice: str,
    product_category: str,
    item_count: int,
    price: float,
    freight_value: float,
    purchase_date_str: str,
    estimated_delivery_date_str: str,
    custom_distance: float
) -> Tuple[str, str, str]:
    """
    Callback generating real ML prediction results based on real Olist features.
    """
    c_state = customer_state_choice if len(customer_state_choice) == 2 else customer_state_choice.split(" - ")[0]
    s_state = seller_state_choice if len(seller_state_choice) == 2 else seller_state_choice.split(" - ")[0]

    dist_arg = float(custom_distance) if custom_distance and custom_distance > 0 else None

    result = predictor.predict(
        customer_state=c_state,
        seller_state=s_state,
        product_category=product_category,
        item_count=int(item_count),
        price=float(price),
        freight_value=float(freight_value),
        purchase_date=purchase_date_str,
        estimated_delivery_date=estimated_delivery_date_str if estimated_delivery_date_str else None,
        distance_km=dist_arg
    )

    days = result["predicted_delivery_days"]
    arr_date = result["estimated_delivery_date"]
    window = result["delivery_window"]
    route = result["route_summary"]
    risk = result["risk_analysis"]
    order = result["order_details"]

    # 1. Primary Highlight Card
    badge_color = "#10B981" if risk["risk_level"] == "Low" else ("#F59E0B" if risk["risk_level"] == "Moderate" else "#EF4444")

    summary_html = f"""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 24px; color: white; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">XGBoost Real Olist Model (R²: 0.400, MAE: 4.49d)</span>
            <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">{risk["risk_level"]} Transit Risk</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 3.5rem; font-weight: 800; color: #38bdf8; line-height: 1;">{days:.1f}</span>
            <span style="font-size: 1.5rem; font-weight: 600; color: #cbd5e1;">Days Delivery Time</span>
        </div>
        <div style="font-size: 1.15rem; color: #f8fafc; font-weight: 500; margin-bottom: 8px;">
            📅 Expected Delivery Date: <strong style="color: #67e8f9;">{arr_date}</strong>
        </div>
        <div style="font-size: 0.95rem; color: #94a3b8;">
            Estimated Delivery Window: <strong>{window['earliest_date']}</strong> ({window['min_days']}d) &mdash; <strong>{window['latest_date']}</strong> ({window['max_days']}d)
        </div>
    </div>
    """

    # 2. Transit Metrics Cards
    same_state_badge = "🟢 Same-State Route" if route["same_state"] else "🟠 Inter-State Route"

    metrics_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;">
        <div style="background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">GEOLOCATION TRANSIT DISTANCE</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin-top: 4px;">{route['distance_km']:.0f} km</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">{s_state} ➔ {c_state} ({same_state_badge})</div>
        </div>
        <div style="background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">ORDER DETAILS & SLA</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin-top: 4px;">R$ {order['freight_value']:.2f} Freight</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">{order['item_count']} item(s) &bull; R$ {order['price']:.2f} &bull; SLA gap: {order['estimated_delivery_gap_days']:.0f}d</div>
        </div>
    </div>
    """

    # 3. Logistics Factors Breakdown
    if risk["risk_factors"]:
        factors_items = "".join([f"<li style='margin-bottom: 4px;'>{f}</li>" for f in risk["risk_factors"]])
    else:
        factors_items = "<li>Optimal intra-state transit corridor &mdash; standard delivery timeline.</li>"

    factors_html = f"""
    <div style="background: #0f172a; padding: 16px; border-radius: 12px; border: 1px solid #1e293b; margin-top: 10px;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;">DATASET FEATURES APPLIED:</div>
        <div style="font-size: 0.85rem; color: #38bdf8; margin-bottom: 8px;">Product Category: <strong>{order['product_category']}</strong> &bull; Purchase Date: <strong>{order['purchase_date']}</strong></div>
        <div style="font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;">IDENTIFIED ROUTE FACTORS:</div>
        <ul style="margin: 0; padding-left: 20px; color: #94a3b8; font-size: 0.9rem;">
            {factors_items}
        </ul>
    </div>
    """

    return summary_html, metrics_html, factors_html


# Modern Gradio UI
custom_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="Olist Delivery Time Prediction System") as demo:
    gr.Markdown(
        """
        # 📦 Olist Brazilian E-Commerce Delivery Time Prediction
        ### Machine Learning Model Trained Exclusively on Real Olist Dataset Columns
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("#### 📍 Route & Product Parameters")
            with gr.Row():
                customer_state_in = gr.Dropdown(
                    choices=[c[0] for c in STATE_CHOICES],
                    value="SP - São Paulo (SP)",
                    label="Customer Destination State (customer_state)"
                )
                seller_state_in = gr.Dropdown(
                    choices=[c[0] for c in STATE_CHOICES],
                    value="SP - São Paulo (SP)",
                    label="Seller Origin State (seller_state)"
                )

            with gr.Row():
                product_category_in = gr.Dropdown(
                    choices=categories,
                    value="bed_bath_table" if "bed_bath_table" in categories else categories[0],
                    label="Product Category (product_category_name_english)"
                )
                item_count_in = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Order Item Count"
                )

            gr.Markdown("#### 💰 Price & Freight Value")
            with gr.Row():
                price_in = gr.Slider(
                    minimum=5.0,
                    maximum=1500.0,
                    value=99.0,
                    step=5.0,
                    label="Order Price (BRL R$)"
                )
                freight_in = gr.Slider(
                    minimum=5.0,
                    maximum=250.0,
                    value=18.5,
                    step=0.5,
                    label="Freight Shipping Value (BRL R$)"
                )

            gr.Markdown("#### 📅 Temporal & SLA Information")
            with gr.Row():
                purchase_date_in = gr.Textbox(
                    value=datetime.now().strftime("%Y-%m-%d"),
                    label="Purchase Date (order_purchase_timestamp)",
                    placeholder="2026-10-15"
                )
                estimated_date_in = gr.Textbox(
                    value=(datetime.now() + timedelta(days=24)).strftime("%Y-%m-%d"),
                    label="Estimated Delivery SLA (order_estimated_delivery_date)",
                    placeholder="2026-11-08"
                )

            with gr.Accordion("⚙️ Geolocation Transit Distance (Optional Override)", open=False):
                custom_dist_in = gr.Number(
                    value=0.0,
                    label="Haversine Distance in km (0 = Auto-calculate from state geolocation coordinates)"
                )

            predict_btn = gr.Button("Calculate Real Delivery Prediction", variant="primary", size="lg")

            gr.Markdown("#### 💡 Quick Test Scenarios")
            with gr.Row():
                scenario1_btn = gr.Button("🏢 Fast Local (SP ➔ SP)", size="sm")
                scenario2_btn = gr.Button("🛋️ SP ➔ RJ Inter-State", size="sm")
                scenario3_btn = gr.Button("🌴 SP ➔ BA Long Distance", size="sm")
                scenario4_btn = gr.Button("🚢 SP ➔ Amazonas Cross-Country", size="sm")

        with gr.Column(scale=5):
            gr.Markdown("#### ⏱️ Prediction & Delivery Intelligence")
            summary_out = gr.HTML()
            metrics_out = gr.HTML()
            factors_out = gr.HTML()

    # Bind button click
    predict_btn.click(
        fn=format_prediction,
        inputs=[
            customer_state_in,
            seller_state_in,
            product_category_in,
            item_count_in,
            price_in,
            freight_in,
            purchase_date_in,
            estimated_date_in,
            custom_dist_in
        ],
        outputs=[summary_out, metrics_out, factors_out]
    )

    # Scenarios logic
    scenario1_btn.click(
        fn=lambda: ("SP - São Paulo (SP)", "SP - São Paulo (SP)", "bed_bath_table", 1, 65.0, 14.0, "2026-06-15", "2026-07-08", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario2_btn.click(
        fn=lambda: ("RJ - Rio de Janeiro (RJ)", "SP - São Paulo (SP)", "furniture_decor", 1, 150.0, 28.0, "2026-05-10", "2026-06-02", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario3_btn.click(
        fn=lambda: ("BA - Bahia (BA)", "SP - São Paulo (SP)", "computers_accessories", 2, 280.0, 48.0, "2026-01-20", "2026-02-15", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario4_btn.click(
        fn=lambda: ("AM - Amazonas (AM)", "SP - São Paulo (SP)", "sports_leisure", 1, 120.0, 68.0, "2026-08-20", "2026-09-18", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    # Initial load trigger
    demo.load(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, item_count_in, price_in, freight_in, purchase_date_in, estimated_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=custom_theme, share=False)
