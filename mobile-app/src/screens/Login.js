import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { loginUser, DUMMY_USERS, initializeDummyUsers } from '../firebase';

function Login({ onLogin }) {
  const [selectedUser, setSelectedUser] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Initialize dummy users on component mount
  useEffect(() => {
    initializeDummyUsers().catch(console.error);
  }, []);

  const handleSubmit = async () => {
    if (!selectedUser) {
      setError('Please select a user');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Find the selected user's email
      const userConfig = DUMMY_USERS.find(u => u.displayName === selectedUser);
      if (!userConfig) {
        throw new Error('User configuration not found');
      }

      // Login with Firebase
      const user = await loginUser(userConfig.email, userConfig.password);
      
      // Pass user info to parent
      onLogin({
        uid: user.uid,
        email: user.email,
        displayName: userConfig.displayName,
        username: userConfig.displayName
      });
    } catch (err) {
      console.error('Login error:', err);
      if (err.code === 'auth/user-not-found') {
        setError('User not found. Initializing users...');
        await initializeDummyUsers();
        setError('Please try again');
      } else {
        setError('Login failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Get display names from Firebase config
  const dummyUsers = DUMMY_USERS.map(user => user.displayName);

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.loginBox}>
          <View style={styles.loginHeader}>
            <Image 
              source={require('../../assets/logo.png')} 
              style={styles.logo}
              resizeMode="contain"
            />
            <Text style={styles.title}>Welcome to KapdaAI</Text>
            <Text style={styles.subtitle}>AI-Powered Clothing Detection & Virtual Try-On</Text>
          </View>

          <View style={styles.form}>
            <Text style={styles.label}>Select User Account</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={selectedUser}
                onValueChange={(itemValue) => setSelectedUser(itemValue)}
                enabled={!loading}
                style={styles.picker}
              >
                <Picker.Item label="-- Choose a user --" value="" />
                {dummyUsers.map(user => (
                  <Picker.Item key={user} label={user} value={user} />
                ))}
              </Picker>
            </View>

            {error ? (
              <View style={styles.errorContainer}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            ) : null}

            <TouchableOpacity
              style={[styles.loginButton, (!selectedUser || loading) && styles.loginButtonDisabled]}
              onPress={handleSubmit}
              disabled={loading || !selectedUser}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.loginButtonText}>Login</Text>
              )}
            </TouchableOpacity>
          </View>

          <View style={styles.footer}>
            <Text style={styles.footerText}>This is a demo application with pre-configured accounts.</Text>
            <Text style={styles.footerText}>No password required for testing.</Text>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  keyboardView: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  loginBox: {
    backgroundColor: '#2a2a2a',
    borderRadius: 12,
    padding: 30,
    maxWidth: 400,
    width: '100%',
    alignSelf: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  loginHeader: {
    alignItems: 'center',
    marginBottom: 30,
  },
  logo: {
    width: 80,
    height: 80,
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
  },
  form: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 10,
  },
  pickerContainer: {
    backgroundColor: '#3a3a3a',
    borderRadius: 8,
    marginBottom: 20,
    overflow: 'hidden',
  },
  picker: {
    color: '#fff',
    height: 50,
  },
  errorContainer: {
    backgroundColor: '#ff444455',
    padding: 10,
    borderRadius: 8,
    marginBottom: 20,
  },
  errorText: {
    color: '#ff4444',
    textAlign: 'center',
  },
  loginButton: {
    backgroundColor: '#4CAF50',
    paddingVertical: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  loginButtonDisabled: {
    backgroundColor: '#4CAF5066',
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  footer: {
    alignItems: 'center',
  },
  footerText: {
    color: '#666',
    fontSize: 14,
    marginBottom: 5,
  },
});

export default Login;