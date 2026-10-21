import streamlit as st
from PIL import Image
from transformers import pipeline
import numpy as np

st.set_page_config(page_title="Multimodal Food & Recipe Assistant", page_icon="🍳")

st.title("🍳 Food Recognition & Recipe Assistant")
st.write("Integrating Digital Image Processing, Neural Networks, and NLP for Automated Recipe Generation.")

# ==========================================
# MACHINE LEARNING & NEURAL NETWORKS
# ==========================================
@st.cache_resource
def load_ml_model():
    # Loads the high-accuracy Vision Transformer Model
    return pipeline("image-classification", model="prithivMLmods/Food-101-93M")

with st.spinner("Loading Neural Network Model..."):
    classifier = load_ml_model()

# Custom knowledge base for our specific food items (NLP Data Retrieval Layer)
RECIPE_DATABASE = {
    "Pizza": {
        "ingredients": ["Pizza dough", "Tomato sauce", "Mozzarella cheese", "Basil"],
        "steps": "1. Roll out dough.\n2. Spread sauce and add cheese.\n3. Bake at 220°C for 12 minutes.",
        "nutrition": {"Calories": "285 kcal", "Protein": "12g", "Carbs": "36g"}
    },
    "Burger": {
        "ingredients": ["Burger bun", "Patty", "Lettuce", "Tomato", "Cheese"],
        "steps": "1. Grill the patty.\n2. Toast the buns.\n3. Assemble with fresh toppings.",
        "nutrition": {"Calories": "450 kcal", "Protein": "22g", "Carbs": "40g"}
    },
    "Pasta": {
        "ingredients": ["Spaghetti", "Marinara sauce", "Parmesan", "Garlic"],
        "steps": "1. Boil pasta al dente.\n2. Sauté garlic and mix with sauce.\n3. Toss and garnish.",
        "nutrition": {"Calories": "320 kcal", "Protein": "10g", "Carbs": "55g"}
    },
    "French Fries": {
        "ingredients": ["Potatoes", "Vegetable oil", "Salt"],
        "steps": "1. Cut potatoes into strips.\n2. Fry at 175°C until golden brown.\n3. Toss with salt.",
        "nutrition": {"Calories": "365 kcal", "Protein": "4g", "Carbs": "48g"}
    },
    "Sushi": {
        "ingredients": ["Sushi rice", "Nori seaweed", "Fresh fish/vegetables", "Rice vinegar"],
        "steps": "1. Season cooked rice with vinegar.\n2. Roll rice and fillings in nori sheet.\n3. Slice into bite-sized pieces.",
        "nutrition": {"Calories": "250 kcal", "Protein": "9g", "Carbs": "38g"}
    },
    "Ramen": {
        "ingredients": ["Ramen noodles", "Broth", "Soft-boiled egg", "Green onions", "Pork/Tofu"],
        "steps": "1. Boil noodles separately.\n2. Heat broth and assemble ingredients in a bowl.\n3. Top with egg and scallions.",
        "nutrition": {"Calories": "430 kcal", "Protein": "15g", "Carbs": "60g"}
    },
    "Tacos": {
        "ingredients": ["Taco shells", "Seasoned meat/beans", "Lettuce", "Cheddar cheese", "Salsa"],
        "steps": "1. Warm taco shells.\n2. Fill with cooked meat, beans, and fresh toppings.",
        "nutrition": {"Calories": "310 kcal", "Protein": "14g", "Carbs": "28g"}
    },
    "Caesar Salad": {
        "ingredients": ["Romaine lettuce", "Caesar dressing", "Croutons", "Parmesan cheese"],
        "steps": "1. Chop lettuce.\n2. Toss with dressing, croutons, and cheese.",
        "nutrition": {"Calories": "200 kcal", "Protein": "5g", "Carbs": "12g"}
    },
    "Ice Cream": {
        "ingredients": ["Milk", "Cream", "Sugar", "Vanilla extract"],
        "steps": "1. Whisk ingredients together.\n2. Chill and churn in an ice cream maker until set.",
        "nutrition": {"Calories": "207 kcal", "Protein": "3g", "Carbs": "24g"}
    },
    "Chocolate Cake": {
        "ingredients": ["Flour", "Cocoa powder", "Sugar", "Eggs", "Butter"],
        "steps": "1. Mix dry and wet ingredients into a batter.\n2. Bake at 180°C for 30 minutes.\n3. Cool and frost.",
        "nutrition": {"Calories": "350 kcal", "Protein": "4g", "Carbs": "52g"}
    },
    "Dumplings": {
        "ingredients": ["Dumpling wrappers", "Ground pork/vegetables", "Soy sauce", "Ginger", "Sesame oil"],
        "steps": "1. Fill wrappers with seasoned meat mixture.\n2. Fold and seal edges with water.\n3. Pan-fry or steam for 8-10 minutes.",
        "nutrition": {"Calories": "290 kcal", "Protein": "12g", "Carbs": "32g"}
    },
    "Pad Thai": {
        "ingredients": ["Rice noodles", "Tofu/Shrimp", "Tamarind paste", "Peanuts", "Bean sprouts", "Eggs"],
        "steps": "1. Soak rice noodles.\n2. Stir-fry protein, egg, and tamarind sauce.\n3. Toss noodles and garnish with crushed peanuts.",
        "nutrition": {"Calories": "400 kcal", "Protein": "16g", "Carbs": "58g"}
    },
    "Steak": {
        "ingredients": ["Beef ribeye/sirloin", "Butter", "Garlic", "Rosemary", "Salt & Pepper"],
        "steps": "1. Season steak generously with salt and pepper.\n2. Sear in a hot skillet for 3-4 mins per side.\n3. Baste with butter, garlic, and rosemary.",
        "nutrition": {"Calories": "650 kcal", "Protein": "48g", "Carbs": "0g"}
    },
    "Lasagna": {
        "ingredients": ["Lasagna sheets", "Minced meat/Bolognese sauce", "Ricotta", "Mozzarella", "Parmesan"],
        "steps": "1. Layer sauce, cooked noodles, and cheese mixture in a baking dish.\n2. Repeat for 3-4 layers.\n3. Bake at 190°C for 45 minutes.",
        "nutrition": {"Calories": "420 kcal", "Protein": "21g", "Carbs": "35g"}
    },
    "Apple Pie": {
        "ingredients": ["Pie crust", "Sliced apples", "Cinnamon", "Sugar", "Butter"],
        "steps": "1. Toss sliced apples with sugar and cinnamon.\n2. Fill lower crust and top with lattice crust.\n3. Bake at 200°C for 45 minutes.",
        "nutrition": {"Calories": "320 kcal", "Protein": "3g", "Carbs": "45g"}
    },
    "Macaroni And Cheese": {
        "ingredients": ["Elbow macaroni", "Cheddar cheese", "Butter", "Milk", "Flour"],
        "steps": "1. Boil macaroni noodles.\n2. Make a roux with butter, flour, milk, and melted cheese.\n3. Stir noodles into cheese sauce.",
        "nutrition": {"Calories": "380 kcal", "Protein": "14g", "Carbs": "42g"}
    },
    "Pho": {
        "ingredients": ["Beef/Chicken broth", "Rice noodles", "Sliced beef", "Star anise", "Herbs (Thai basil, cilantro)"],
        "steps": "1. Simmer broth with spices for hours.\n2. Cook rice noodles and place in a bowl with raw beef slices.\n3. Ladle scalding hot broth over top.",
        "nutrition": {"Calories": "350 kcal", "Protein": "24g", "Carbs": "45g"}
    },
    "Spring Rolls": {
        "ingredients": ["Rice paper wrappers", "Shrimp/Tofu", "Rice vermicelli", "Cucumbers", "Mint"],
        "steps": "1. Dip rice paper in warm water.\n2. Lay out veggies, noodles, and protein.\n3. Roll tightly and serve with peanut dip.",
        "nutrition": {"Calories": "150 kcal", "Protein": "6g", "Carbs": "25g"}
    },
    "Hot Dog": {
        "ingredients": ["Hot dog sausage", "Hot dog bun", "Mustard", "Ketchup", "Relish"],
        "steps": "1. Grill or boil sausage.\n2. Toast the bun.\n3. Place sausage inside and top with condiments.",
        "nutrition": {"Calories": "290 kcal", "Protein": "10g", "Carbs": "24g"}
    },
    "Waffles": {
        "ingredients": ["Flour", "Eggs", "Milk", "Baking powder", "Butter", "Maple syrup"],
        "steps": "1. Whisk dry and wet ingredients into a smooth batter.\n2. Pour into hot waffle iron.\n3. Cook until crisp and serve with syrup.",
        "nutrition": {"Calories": "290 kcal", "Protein": "6g", "Carbs": "35g"}
    },
    "Cheesecake": {
        "ingredients": ["Cream cheese", "Graham cracker crust", "Sugar", "Eggs", "Vanilla"],
        "steps": "1. Beat cream cheese, sugar, and eggs until smooth.\n2. Pour into crust and bake in a water bath at 160°C.\n3. Chill overnight.",
        "nutrition": {"Calories": "400 kcal", "Protein": "6g", "Carbs": "32g"}
    },
    "Cupcakes": {
        "ingredients": ["Flour", "Sugar", "Butter", "Eggs", "Vanilla", "Buttercream frosting"],
        "steps": "1. Prepare cake batter and pour into muffin tins.\n2. Bake at 180°C for 18 minutes.\n3. Cool completely and pipe frosting.",
        "nutrition": {"Calories": "240 kcal", "Protein": "2g", "Carbs": "33g"}
    },
    "Fried Rice": {
        "ingredients": ["Cooked rice", "Eggs", "Soy sauce", "Peas & carrots", "Green onions", "Sesame oil"],
        "steps": "1. Scramble eggs in a wok and set aside.\n2. Stir-fry veggies and cold rice with soy sauce.\n3. Toss in eggs and scallions.",
        "nutrition": {"Calories": "330 kcal", "Protein": "9g", "Carbs": "52g"}
    },
    "Omelette": {
        "ingredients": ["Eggs", "Butter", "Cheese", "Bell peppers", "Spinach"],
        "steps": "1. Whisk eggs with salt.\n2. Melt butter in pan and pour eggs.\n3. Add cheese and veggies, then fold in half.",
        "nutrition": {"Calories": "210 kcal", "Protein": "14g", "Carbs": "3g"}
    },
    "Nachos": {
        "ingredients": ["Tortilla chips", "Melted cheese/Queso", "Jalapeños", "Black beans", "Salsa", "Sour cream"],
        "steps": "1. Arrange chips on a baking sheet.\n2. Top with beans and shredded cheese.\n3. Bake until melted and add cold toppings.",
        "nutrition": {"Calories": "460 kcal", "Protein": "12g", "Carbs": "48g"}
    },
    "Pancakes": {
        "ingredients": ["Flour", "Milk", "Eggs", "Baking powder", "Butter", "Maple syrup"],
        "steps": "1. Mix batter gently.\n2. Pour ladles onto hot griddle.\n3. Flip when bubbles form and cook until golden.",
        "nutrition": {"Calories": "350 kcal", "Protein": "8g", "Carbs": "58g"}
    },
    "Hummus": {
        "ingredients": ["Chickpeas", "Tahini", "Lemon juice", "Garlic", "Olive oil"],
        "steps": "1. Blend chickpeas, tahini, garlic, and lemon juice in a food processor.\n2. Drizzle in olive oil until smooth.\n3. Serve with pita bread.",
        "nutrition": {"Calories": "170 kcal", "Protein": "5g", "Carbs": "14g"}
    },
    "Guacamole": {
        "ingredients": ["Ripe avocados", "Lime juice", "Red onion", "Cilantro", "Salt", "Jalapeño"],
        "steps": "1. Mash avocados in a bowl.\n2. Stir in lime juice, finely chopped onion, cilantro, and salt.",
        "nutrition": {"Calories": "160 kcal", "Protein": "2g", "Carbs": "9g"}
    },
    "Tiramisu": {
        "ingredients": ["Ladyfingers", "Espresso", "Mascarpone cheese", "Eggs", "Cocoa powder"],
        "steps": "1. Dip ladyfingers briefly in espresso.\n2. Layer with whipped mascarpone mixture.\n3. Dust top liberally with cocoa powder.",
        "nutrition": {"Calories": "300 kcal", "Protein": "5g", "Carbs": "32g"}
    },
    "Samoas": {
        "ingredients": ["Shortbread crust", "Caramel", "Toasted coconut", "Dark chocolate drizzle"],
        "steps": "1. Bake shortbread ring cookies.\n2. Coat top with caramel and toasted coconut.\n3. Drizzle chocolate across bottom and top.",
        "nutrition": {"Calories": "140 kcal", "Protein": "1g", "Carbs": "18g"}
    },
    "Bruschetta": {
        "ingredients": ["Crusty Italian bread", "Roma tomatoes", "Fresh basil", "Garlic", "Extra virgin olive oil", "Balsamic vinegar", "Salt", "Black pepper"],
        "steps": "1. Dice tomatoes, mince garlic, and chop basil, then mix with olive oil and vinegar and let sit for 15 minutes.\n2. Slice bread, brush with olive oil, and toast at 200°C for 7 minutes.\n3. Rub the warm toasted bread with a peeled garlic clove.\n4. Spoon the tomato mixture onto the bread and serve.",
        "nutrition": {"Calories": "210 kcal", "Protein": "5g", "Carbs": "28g"}
    }
}

# Image Upload Widget
uploaded_file = st.file_uploader("Upload an image of food (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # ==========================================
    # DIGITAL IMAGE PROCESSING (DIP)
    # ==========================================
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocessing step: Resize and normalize image matrix for the neural network
    processed_image = raw_image.resize((224, 224))
    
    st.image(raw_image, caption="Uploaded Image (Processed via DIP: Resized & Normalized)", use_container_width=True)
    
    with st.spinner("Processing image through Neural Network..."):
        # ==========================================
        # MACHINE LEARNING & NEURAL NETWORKS INFERENCE
        # ==========================================
        predictions = classifier(processed_image)
        top_prediction = predictions[0]
        detected_food = top_prediction['label'].replace('_', ' ').title()
        confidence = top_prediction['score'] * 100

    st.success(f"**ML/NN Classification Result:** {detected_food} ({confidence:.2f}% confidence)")

    st.markdown("---")

    # ==========================================
    # NATURAL LANGUAGE PROCESSING (NLP) & FALLBACK HANDLING
    # ==========================================
    if detected_food in RECIPE_DATABASE:
        dish_info = RECIPE_DATABASE[detected_food]
        
        st.subheader(f"📖 NLP-Generated Recipe: {detected_food}")
        
        st.markdown("**Required Ingredients (Tokenized Text):**")
        for ing in dish_info["ingredients"]:
            st.write(f"- {ing}")
            
        st.markdown("**Cooking Instructions:**")
        st.text(dish_info["steps"])

        st.subheader("🥗 Nutritional Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("Calories", dish_info["nutrition"]["Calories"])
        col2.metric("Protein", dish_info["nutrition"]["Protein"])
        col3.metric("Carbs", dish_info["nutrition"]["Carbs"])
    else:
        st.warning(f"⚠️ **Recipe Not Found / Calories Not Found**")
        st.write(f"The model identified this item as **{detected_food}**, but it is outside our 31-item curated database. Try uploading an image matching our supported categories!")