import os
import tensorflow as tf
from optimized_model import PestDetectionModel
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import json
from datetime import datetime
import matplotlib.pyplot as plt

class TrainingPipeline:
    def __init__(self, train_dirs, test_dirs, output_dir='pest_model'):
        self.train_dirs = train_dirs
        self.test_dirs = test_dirs
        self.output_dir = output_dir
        self.input_shape = (380, 380, 3)  # Optimal for EfficientNetB4
        self.batch_size = 8  # Smaller batch size for larger model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_datasets(self):
        """Prepare training and testing datasets"""
        # Load and combine datasets
        train_ds = []
        test_ds = []
        all_classes = set()
        
        # Process training directories
        for train_dir in self.train_dirs:
            if os.path.exists(train_dir):
                ds = tf.keras.utils.image_dataset_from_directory(
                    train_dir,
                    label_mode='int',
                    image_size=self.input_shape[:2],
                    batch_size=self.batch_size,
                    shuffle=True
                )
                train_ds.append(ds)
                all_classes.update(ds.class_names)
        
        # Process test directories
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                ds = tf.keras.utils.image_dataset_from_directory(
                    test_dir,
                    label_mode='int',
                    image_size=self.input_shape[:2],
                    batch_size=self.batch_size
                )
                test_ds.append(ds)
        
        # Combine datasets
        if not train_ds:
            raise ValueError("No training data found")
        
        self.train_ds = train_ds[0].concatenate(train_ds[1]) if len(train_ds) > 1 else train_ds[0]
        self.test_ds = test_ds[0].concatenate(test_ds[1]) if len(test_ds) > 1 else test_ds[0]
        self.class_names = sorted(list(all_classes))
        self.num_classes = len(self.class_names)
        
        # Configure datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(AUTOTUNE)
        
        return self.train_ds, self.test_ds, self.num_classes
    
    def setup_callbacks(self, phase):
        """Setup training callbacks for each phase"""
        return [
            ModelCheckpoint(
                os.path.join(self.output_dir, f'best_model_phase{phase}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
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
                min_lr=1e-6,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.output_dir, f'training_log_phase{phase}.csv'),
                append=True
            )
        ]

    def _save_training_visualizations(self, history1, history2):
        """Generate and save training visualizations"""
        # Combine histories
        combined_acc = history1.history['accuracy'] + history2.history['accuracy']
        combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        combined_loss = history1.history['loss'] + history2.history['loss']
        combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        # Plot accuracy
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(combined_acc, label='Training Accuracy', linewidth=2)
        plt.plot(combined_val_acc, label='Validation Accuracy', linewidth=2)
        plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Phase Change')
        plt.title('Model Accuracy Over Training Phases', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(combined_loss, label='Training Loss', linewidth=2)
        plt.plot(combined_val_loss, label='Validation Loss', linewidth=2)
        plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Phase Change')
        plt.title('Model Loss Over Training Phases', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_model_metrics(self, model, history1, history2):
        """Save model metrics and performance summary"""
        metrics = {
            'phase1': {
                'final_accuracy': float(history1.history['accuracy'][-1]),
                'final_val_accuracy': float(history1.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history1.history['val_accuracy'])),
                'final_loss': float(history1.history['loss'][-1]),
                'epochs_trained': len(history1.history['accuracy'])
            },
            'phase2': {
                'final_accuracy': float(history2.history['accuracy'][-1]),
                'final_val_accuracy': float(history2.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history2.history['val_accuracy'])),
                'final_loss': float(history2.history['loss'][-1]),
                'epochs_trained': len(history2.history['accuracy'])
            },
            'model_info': {
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'architecture': 'EfficientNetB4',
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'class_names': self.class_names
            }
        }
        
        with open(os.path.join(self.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    def train(self, epochs_per_phase=10):
        """Train the model in two phases: top layers and fine-tuning"""
        print("\n=== Starting Training Pipeline ===")
        model = PestDetectionModel(self.num_classes, self.input_shape)
        
        # Phase 1: Train only top layers
        print("\n=== Phase 1: Training Top Layers ===")
        print("Freezing EfficientNetB4 layers...")
        model.compile_model(learning_rate=1e-3)
        
        callbacks_phase1 = self.setup_callbacks(phase=1)
        
        print("\nPhase 1: Training top layers...")
        history1 = model.model.fit(
            self.train_ds,
            validation_data=self.test_ds,
            epochs=epochs_per_phase,
            callbacks=callbacks_phase1,
            verbose=1
        )

        # Save Phase 1 results
        model.model.save_weights(os.path.join(self.output_dir, 'phase1_final.h5'))
        
        # Phase 2: Fine-tuning
        print("\n=== Phase 2: Fine-tuning EfficientNetB4 ===")
        print("Unfreezing EfficientNetB4 layers...")
        model.unfreeze_and_compile(learning_rate=1e-4)
        
        callbacks_phase2 = self.setup_callbacks(phase=2)
        
        history2 = model.model.fit(
            self.train_ds,
            validation_data=self.test_ds,
            epochs=epochs_per_phase,
            callbacks=callbacks_phase2,
            verbose=1
        )

        # Save final model and generate visualizations
        print("\n=== Generating Final Results ===")
        
        # Save final model
        model.model.save_weights(os.path.join(self.output_dir, 'final_model.h5'))
        
        # Generate and save training visualizations
        self._save_training_visualizations(history1, history2)
        
        # Generate and save model metrics
        self._save_model_metrics(model, history1, history2)
        
        print("\n=== Training Complete ===")
        return history1, history2

def main():
    # Setup directories
    train_dirs = ['dataset/train', 'pest/train']
    test_dirs = ['dataset/test', 'pest/test']
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(train_dirs, test_dirs)
    train_ds, test_ds, num_classes = pipeline.prepare_datasets()
    
    print(f"Starting training with {num_classes} classes...")
    history1, history2 = pipeline.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()