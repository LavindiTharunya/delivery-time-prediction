"""
Interactive Delivery Time Prediction Web Application (Gradio).
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
    price: float,
    freight_value: float,
    purchase_date_str: str,
    custom_distance: float
) -> Tuple[str, str, str, str, str]:
    """
    Callback function that generates prediction results and formatted HTML cards.
    """
    c_state = customer_state_choice if len(customer_state_choice) == 2 else customer_state_choice.split(" - ")[0]
    s_state = seller_state_choice if len(seller_state_choice) == 2 else seller_state_choice.split(" - ")[0]

    dist_arg = float(custom_distance) if custom_distance and custom_distance > 0 else None

    result = predictor.predict(
        customer_state=c_state,
        seller_state=s_state,
        price=price,
        freight_value=freight_value,
        product_category=product_category,
        purchase_date=purchase_date_str,
        distance_km=dist_arg
    )

    days = result["predicted_delivery_days"]
    arr_date = result["estimated_delivery_date"]
    window = result["delivery_window"]
    route = result["route_summary"]
    risk = result["risk_analysis"]

    # 1. Primary Highlight Card
    badge_color = "#10B981" if risk["risk_level"] == "Low" else ("#F59E0B" if risk["risk_level"] == "Moderate" else "#EF4444")
    
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 16px; padding: 24px; color: white; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">XGBoost Enhanced v3 Estimate</span>
            <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">{risk["risk_level"]} Risk</span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 3.5rem; font-weight: 800; color: #38bdf8; line-height: 1;">{days:.1f}</span>
            <span style="font-size: 1.5rem; font-weight: 600; color: #cbd5e1;">Days</span>
        </div>
        <div style="font-size: 1.15rem; color: #f8fafc; font-weight: 500; margin-bottom: 8px;">
            📅 Expected Arrival: <strong style="color: #67e8f9;">{arr_date}</strong>
        </div>
        <div style="font-size: 0.95rem; color: #94a3b8;">
            Estimated Delivery Window: <strong>{window['earliest_date']}</strong> ({window['min_days']}d) &mdash; <strong>{window['latest_date']}</strong> ({window['max_days']}d)
        </div>
    </div>
    """

    # 2. Transit Metrics Cards
    same_state_badge = "🟢 Same-State Delivery" if route["same_state"] else "🟠 Inter-State Route"
    holiday_badge = "🎉 Near National Holiday" if risk["near_holiday"] else "✅ Standard Operating Period"

    metrics_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;">
        <div style="background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">ROUTE DISTANCE</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin-top: 4px;">{route['distance_km']:.0f} km</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">{s_state} ➔ {c_state} ({same_state_badge})</div>
        </div>
        <div style="background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">CALENDAR IMPACT</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9; margin-top: 4px;">{holiday_badge}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">Purchase Date: {purchase_date_str}</div>
        </div>
    </div>
    """

    # 3. Risk Factors Breakdown
    if risk["risk_factors"]:
        factors_items = "".join([f"<li style='margin-bottom: 4px;'>{f}</li>" for f in risk["risk_factors"]])
    else:
        factors_items = "<li>Optimal routing conditions &mdash; standard transit timeline.</li>"

    factors_html = f"""
    <div style="background: #0f172a; padding: 16px; border-radius: 12px; border: 1px solid #1e293b; margin-top: 10px;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px;">TRANSIT DRIVING FACTORS:</div>
        <ul style="margin: 0; padding-left: 20px; color: #94a3b8; font-size: 0.9rem;">
            {factors_items}
        </ul>
    </div>
    """

    return summary_html, metrics_html, factors_html


# Build custom modern UI with Gradio Blocks
custom_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="E-Commerce Delivery Time Prediction System", theme=custom_theme) as demo:
    gr.Markdown(
        """
        # 📦 E-Commerce Delivery Time Prediction System
        ### Real-Time Machine Learning Logistics Estimator for Brazilian E-Commerce Orders
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("#### 📍 Order & Geolocation Parameters")
            with gr.Row():
                customer_state_in = gr.Dropdown(
                    choices=[c[0] for c in STATE_CHOICES],
                    value="SP - São Paulo (SP)",
                    label="Customer Destination State"
                )
                seller_state_in = gr.Dropdown(
                    choices=[c[0] for c in STATE_CHOICES],
                    value="SP - São Paulo (SP)",
                    label="Seller Origin State"
                )

            with gr.Row():
                product_category_in = gr.Dropdown(
                    choices=categories,
                    value="office_furniture" if "office_furniture" in categories else categories[0],
                    label="Product Category"
                )
                purchase_date_in = gr.Textbox(
                    value=datetime.now().strftime("%Y-%m-%d"),
                    label="Purchase Date (YYYY-MM-DD)",
                    placeholder="2026-10-15"
                )

            with gr.Row():
                price_in = gr.Slider(
                    minimum=5.0,
                    maximum=3000.0,
                    value=120.0,
                    step=5.0,
                    label="Order Price (BRL R$)"
                )
                freight_in = gr.Slider(
                    minimum=0.0,
                    maximum=200.0,
                    value=18.5,
                    step=1.0,
                    label="Freight Shipping Fee (BRL R$)"
                )

            with gr.Accordion("⚙️ Advanced Route Distance Override (Optional)", open=False):
                custom_dist_in = gr.Number(
                    value=0.0,
                    label="Custom Haversine Distance in km (0 = Auto-calculate from state centroids)"
                )

            predict_btn = gr.Button("🚀 Calculate Delivery Estimate", variant="primary", size="lg")

            gr.Markdown("#### 💡 Quick Test Scenarios")
            with gr.Row():
                scenario1_btn = gr.Button("🏢 SP Local Same-State", size="sm")
                scenario2_btn = gr.Button("🌴 SP to Bahia (Inter-State)", size="sm")
                scenario3_btn = gr.Button("🚢 SP to Amazonas (Cross-Country)", size="sm")
                scenario4_btn = gr.Button("🛍️ Black Friday Surge (Nov)", size="sm")

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
            price_in,
            freight_in,
            purchase_date_in,
            custom_dist_in
        ],
        outputs=[summary_out, metrics_out, factors_out]
    )

    # Scenarios logic
    scenario1_btn.click(
        fn=lambda: ("SP - São Paulo (SP)", "SP - São Paulo (SP)", "bed_bath_table", 89.90, 12.50, "2026-06-15", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario2_btn.click(
        fn=lambda: ("BA - Bahia (BA)", "SP - São Paulo (SP)", "computers_accessories", 250.0, 32.0, "2026-07-10", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario3_btn.click(
        fn=lambda: ("AM - Amazonas (AM)", "SP - São Paulo (SP)", "sports_leisure", 180.0, 58.0, "2026-08-20", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    scenario4_btn.click(
        fn=lambda: ("RJ - Rio de Janeiro (RJ)", "SP - São Paulo (SP)", "watches_gifts", 340.0, 24.0, "2026-11-25", 0.0),
        outputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in]
    ).then(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

    # Initial load trigger
    demo.load(
        fn=format_prediction,
        inputs=[customer_state_in, seller_state_in, product_category_in, price_in, freight_in, purchase_date_in, custom_dist_in],
        outputs=[summary_out, metrics_out, factors_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
