#!/usr/bin/env python3
"""
Enhanced Pest Detection Training Script with Visualizations
Trains a MobileNetV2 model for pest detection with comprehensive metrics and visualizations
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PestDetectionTrainer:
    def __init__(self, train_dir, test_dir, output_dir='pest_model'):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.input_shape = (224, 224, 3)
        self.batch_size = 32
        self.num_classes = 0
        self.class_names = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self):
        """Prepare and load the dataset"""
        print("📊 Preparing dataset...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load test data
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = self.train_generator.num_classes
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Classes: {self.num_classes}")
        print(f"   Class names: {self.class_names}")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        
        return self.train_generator, self.test_generator
    
    def build_model(self):
        """Build the MobileNetV2 model with custom head"""
        print("🏗️ Building model architecture...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def train_model(self, epochs=20):
        """Train the model with callbacks and monitoring"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, 'pest_model_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.output_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.test_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self):
        """Evaluate the model and generate metrics"""
        print("📊 Evaluating model...")
        
        # Get predictions
        test_predictions = self.model.predict(self.test_generator)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = self.test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=0)
        
        print(f"📈 Test Results:")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Loss: {test_loss:.4f}")
        
        # Classification report
        class_report = classification_report(
            test_true_classes, 
            test_pred_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save classification report
        with open(os.path.join(self.output_dir, 'classification_report.json'), 'w') as f:
            json.dump(class_report, f, indent=2)
        
        return test_accuracy, test_loss, class_report, test_pred_classes, test_true_classes
    
    def plot_training_history(self, history):
        """Plot training history with learning rate"""
        print("📊 Generating training visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pest Detection Model - Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy
        ax1 = axes[0, 0]
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss
        ax2 = axes[0, 1]
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate (if available)
        ax3 = axes[1, 0]
        if 'lr' in history.history:
            ax3.plot(history.history['lr'], linewidth=2, marker='o', color='green')
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
        else:
            # Simulate learning rate schedule
            epochs = len(history.history['accuracy'])
            initial_lr = 0.001
            lr_schedule = [initial_lr * (0.5 ** (i // 3)) for i in range(epochs)]
            ax3.plot(lr_schedule, linewidth=2, marker='o', color='green')
            ax3.set_title('Learning Rate Schedule (Simulated)', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Top-3 Accuracy (if available)
        ax4 = axes[1, 1]
        if 'top_3_accuracy' in history.history:
            ax4.plot(history.history['top_3_accuracy'], label='Top-3 Accuracy', linewidth=2, marker='o')
            ax4.plot(history.history['val_top_3_accuracy'], label='Val Top-3 Accuracy', linewidth=2, marker='s')
            ax4.set_title('Top-3 Accuracy', fontweight='bold')
        else:
            # Calculate top-3 accuracy from predictions
            test_predictions = self.model.predict(self.test_generator)
            top3_acc = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                self.test_generator.classes, test_predictions, k=3
            ).numpy().mean()
            ax4.text(0.5, 0.5, f'Top-3 Accuracy: {top3_acc:.4f}', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14, fontweight='bold')
            ax4.set_title('Top-3 Accuracy', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training history plots saved!")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        print("📊 Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Pest Detection Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix saved!")
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        print("📊 Generating class distribution...")
        
        # Get class counts
        class_counts = self.train_generator.classes
        unique, counts = np.unique(class_counts, return_counts=True)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(self.class_names)), counts, color='skyblue', alpha=0.8)
        plt.title('Training Data Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Pest Classes', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Class distribution plot saved!")
    
    def save_model_info(self, test_accuracy, test_loss, class_report):
        """Save model information and metadata"""
        print("💾 Saving model information...")
        
        # Save class names
        with open(os.path.join(self.output_dir, 'labels.json'), 'w') as f:
            json.dump(self.class_names, f, indent=2)
        
        # Save class mappings
        class_mappings = {name: idx for idx, name in enumerate(self.class_names)}
        with open(os.path.join(self.output_dir, 'class_mappings.json'), 'w') as f:
            json.dump(class_mappings, f, indent=2)
        
        # Save training summary
        summary = {
            'model_architecture': 'MobileNetV2',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'training_samples': self.train_generator.samples,
            'test_samples': self.test_generator.samples,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_parameters': int(self.model.count_params()),
            'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
            'class_report': class_report
        }
        
        with open(os.path.join(self.output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Model information saved!")
    
    def run_training(self, epochs=20):
        """Run complete training pipeline"""
        print("🌱 Starting Pest Detection Training Pipeline")
        print("=" * 50)
        
        # Prepare data
        self.prepare_data()
        
        # Build model
        self.model = self.build_model()
        
        # Plot class distribution
        self.plot_class_distribution()
        
        # Train model
        history = self.train_model(epochs)
        
        # Evaluate model
        test_accuracy, test_loss, class_report, y_pred, y_true = self.evaluate_model()
        
        # Generate visualizations
        self.plot_training_history(history)
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Save model and info
        self.model.save(os.path.join(self.output_dir, 'pest_model_final.h5'))
        self.save_model_info(test_accuracy, test_loss, class_report)
        
        print("=" * 50)
        print("🎉 Training pipeline completed successfully!")
        print(f"📊 Final Test Accuracy: {test_accuracy:.4f}")
        print(f"📊 Final Test Loss: {test_loss:.4f}")
        print(f"💾 Model saved to: {self.output_dir}/")
        
        return history, test_accuracy, test_loss

def main():
    """Main training function"""
    # Setup directories
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Initialize trainer
    trainer = PestDetectionTrainer(train_dir, test_dir)
    
    # Run training
    history, accuracy, loss = trainer.run_training(epochs=20)
    
    print(f"\n✅ Training completed with {accuracy:.4f} accuracy!")

if __name__ == "__main__":
    main()
